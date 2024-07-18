import torch
import numpy as np
import random
import pickle as pk
import config as default_config
from torch.utils.data import Dataset, DataLoader
from pytorch_metric_learning import samplers


class MOSIDataset(Dataset):
    #初始化，加载相关属性
    def __init__(self, type, use_similarity=False, simi_return_mono=False, config=default_config):
        raw_data_path = config.MOSI.path.raw_data_path#'data/MOSI/Processed/unaligned_50.pkl'
        with open(raw_data_path, 'rb') as f:
            self.data = pk.load(f)[type]

        #检查audio_lengths和vision_lengths是否存在，不存在则进行计算
        if 'audio_lengths' not in self.data.keys():
            audio_len = self.data['audio'].shape[1]#音频数据的特征数量
            self.data['audio_lengths'] = [audio_len] * self.data['audio'].shape[0]
        if 'vision_lengths' not in self.data.keys():
            vision_len = self.data['vision'].shape[1]
            self.data['vision_lengths'] = [vision_len] * self.data['vision'].shape[0]

        #设置其他属性
        self.simi_return_mono = simi_return_mono
        self.data['raw_text'] = np.array(self.data['raw_text'])
        self.data['id'] = np.array(self.data['id'])
        self.size = len(self.data['raw_text'])
        self.data['index'] = torch.tensor(range(self.size))
        self.vision_fea_size = self.data['vision'][0].shape
        self.audio_fea_size = self.data['audio'][0].shape
        self.scaled_embedding_averaged = False

        self.__gen_mask()#生成掩码
        if type == 'train' and use_similarity:
            self.__scale()#缩放数据
            self.__gen_cos_matrix()#生成余弦矩阵
        self.use_similarity = use_similarity#是否使用相似度的属性

    #计算了视觉和音频数据中每一行中非零元素的数量。

   #生成了视觉和音频数据的掩码，用于在处理序列数据时忽略某些元素。
    def __gen_mask(self):
        vision_tmp = torch.sum(torch.tensor(self.data['vision']), dim=-1)
        vision_mask = (vision_tmp == 0)

        #遍历 vision_mask 的每一行，并将每一行的第一个元素设置为 False
        for i in range(self.size):
            vision_mask[i][0] = False
        #在最后一个维度上将 vision_mask 的第一列和 vision_mask 本身拼接起来。
        vision_mask = torch.cat((vision_mask[:, 0:1], vision_mask), dim=-1)

        self.data['vision_padding_mask'] = vision_mask
        audio_tmp = torch.sum(torch.tensor(self.data['audio']), dim=-1)
        audio_mask = (audio_tmp == 0)
        for i in range(self.size):
            audio_mask[i][0] = False
        audio_mask = torch.cat((audio_mask[:, 0:1], audio_mask), dim=-1)
        self.data['audio_padding_mask'] = audio_mask

    #创建了相应的填充掩码，以确保模型不会处理填充（零）值。
    #作用：在音频和视觉数据中添加填充列，以便在处理不同长度的序列时保持一致。
    def __pad(self):
        #创建一个大小为 (样本数, 1, 特征维度) 的零张量 PAD。
        PAD = torch.zeros(self.data['vision'].shape[0], 1, self.data['vision'].shape[2])
        #将 PAD 沿着第二个维度（列）与原始视觉数据进行连接，以添加一个填充列。
        self.data['vision'] = np.concatenate((self.data['vision'], PAD), axis=1)
        #创建一个大小为 (样本数, 特征维度) 的全 1 张量 Ones
        Ones = torch.ones(self.data['vision'].shape[0], self.data['vision'].shape[2])
        #将 Ones 的值赋给每个样本的填充部分，以确保填充的部分不影响模型的计算。
        for i in range(len(self.data['vision'])):
            self.data['vision'][i, self.data['vision_lengths'], :] = Ones

        #对音频数据执行相同的填充操作。
        PAD = torch.zeros(self.data['audio'].shape[0], 1, self.data['audio'].shape[2])
        self.data['audio'] = np.concatenate((self.data['audio'], PAD), axis=1)
        Ones = torch.ones(self.data['audio'].shape[0], self.data['audio'].shape[2])
        for i in range(len(self.data['audio'])):
            self.data['audio'][i, self.data['audio_lengths'], :] = Ones

    #对视觉和音频数据进行归一化处理，包括转置维度、计算平均值和替换NaN值等操作。
    def __normalize(self):
        # (num_examples,max_len,feature_dim) -> (max_len, num_examples, feature_dim)
        self.data['vision'] = np.transpose(self.data['vision'], (1, 0, 2))
        self.data['audio'] = np.transpose(self.data['audio'], (1, 0, 2))
        # for visual and audio modality, we average across time
        # here the original data has shape (max_len, num_examples, feature_dim)
        # after averaging they become (1, num_examples, feature_dim)
        self.data['vision'] = np.mean(self.data['vision'], axis=0, keepdims=True)
        self.data['audio'] = np.mean(self.data['audio'], axis=0, keepdims=True)

        # remove possible NaN values
        #将视觉和音频数据中的NaN值（不是一个数字）替换为0。
        self.data['vision'][self.data['vision'] != self.data['vision']] = 0
        self.data['audio'][self.data['audio'] != self.data['audio']] = 0

        self.data['vision'] = np.transpose(self.data['vision'], (1, 0, 2))
        self.data['audio'] = np.transpose(self.data['audio'], (1, 0, 2))

    #用于数据缩放
    def __scale(self):
        #创建副本
        self.scaled_audio = self.data['audio'].copy()
        self.scaled_vision = self.data['vision'].copy()
        self.scaled_text = self.data['text'].copy()
        #对音频和视觉数据进行缩放
        for i in range(self.audio_fea_size[-1]):
            #对于每个音频特征维度，计算最大值和最小值。
            max_num = np.max(self.data['audio'][:, :, i])
            min_num = np.min(self.data['audio'][:, :, i])
            #使用最大值和最小值对音频数据进行缩放，使其范围在 [-1, 1] 之间。
            self.scaled_audio[:, :, i] = (self.data['audio'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        #对视觉数据进行同样操作
        for i in range(self.vision_fea_size[-1]):
            max_num = np.max(self.data['vision'][:, :, i])
            min_num = np.min(self.data['vision'][:, :, i])
            self.scaled_vision[:, :, i] = (self.data['vision'][:, :, i] - min_num) / (max_num - min_num) * 2 - 1
        #转换为pytorch张量
        self.scaled_audio = torch.tensor(self.scaled_audio)
        self.scaled_vision = torch.tensor(self.scaled_vision)
        self.scaled_text = torch.tensor(self.scaled_text)

    #计算余弦相似度矩阵
    def __gen_cos_matrix(self, model=None):
        self.cos_matrix_M = torch.zeros((self.size, self.size))

        self.text_fea = torch.zeros((self.size, self.size))

        #创建了一个 CosineSimilarity 对象 cos，它用于计算余弦相似度。
        cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        if not self.scaled_embedding_averaged:
            # calculated mean
            audio_mean = torch.sum(self.scaled_audio, dim=-2)
            vision_mean = torch.sum(self.scaled_vision, dim=-2)
            text_mean = torch.sum(self.scaled_text, dim=-2)

            for i in range(len(audio_mean)):
                audio_mean[i, :] /= self.data['audio_lengths'][i]
            for i in range(len(vision_mean)):
                vision_mean[i, :] /= self.data['vision_lengths'][i]
            for i in range(len(text_mean)):
                text_mean[i, :] /= 50
        else:
            text_mean, vision_mean, audio_mean = self.scaled_text, self.scaled_vision, self.scaled_audio
            # vision_mean = torch.sum(self.scaled_vision, dim=-2)
            # audio_mean = torch.sum(self.scaled_audio, dim=-2)

            # for i in range(len(vision_mean)):
            #     vision_mean[i, :] /= self.data['vision_lengths'][i]
            # for i in range(len(audio_mean)):
            #     audio_mean[i, :] /= self.data['audio_lengths'][i]

        #余弦相似度矩阵
        self.cos_matrix_M = cos(torch.cat((text_mean, vision_mean, audio_mean), dim=-1).unsqueeze(1),
                                torch.cat((text_mean, vision_mean, audio_mean), dim=-1).unsqueeze(0))

        self.rank_M = torch.zeros(self.cos_matrix_M.shape)

        #对 cos_matrix_M 的每一行进行排序，并将排序后的索引保存在 rank_M 中
        for i in range(len(self.cos_matrix_M)):
            _, self.rank_M[i, :] = torch.sort(self.cos_matrix_M[i, :], descending=True)
        #
        self.M_retrieve = self.__pre_sample(self.rank_M, np.round(self.data['regression_labels']))


    #预处理样本数据，包括生成四种类型的样本列表，并检查每种类型的样本数量是否足够。
    def __pre_sample(self, _rank, _label):
        retrieve = {'ss': [],
                    'sd': [],
                    'ds': [],
                    'dd': [],
                    }
        for i in range(self.size):
            _ss = []
            _sd = []
            _ds = []
            _dd = []
            for j in range(int(self.size)):
                if i == j: continue
                if np.round(_label[i]) == _label[int(_rank[i][j])]:
                    _ss.append(j)
                else:
                    _sd.append(j)
            for j in range(-1, -int(self.size), -1):
                if i == j: continue
                if np.round(_label[i]) == _label[int(_rank[i][j])]:
                    _ds.append(j)
                else:
                    _dd.append(j)
            if len(_ss) < 2 or len(_sd) < 2 or len(_ds) < 2 or len(_dd) < 2:
                print('Unique sample detected, may cause error!')

            #将_ss、_sd、_ds和_dd的前10个元素添加到retrieve字典中对应的列表中。
            retrieve['ss'].append(_ss[:10])
            retrieve['sd'].append(_sd[:10])
            retrieve['ds'].append(_ds[:10])
            retrieve['dd'].append(_dd[:10])
        return retrieve

    #更新矩阵
    def update_matrix(self, T, V, A):
        self.scaled_text = T
        self.scaled_vision = V
        self.scaled_audio = A
        self.scaled_embedding_averaged = True
        self.__gen_cos_matrix()

    def sample(self, _sample_idx):
        if not self.simi_return_mono:
            samples2 = {}
            idx2 = []
            for i in _sample_idx:
                #从 self.M_retrieve 中随机抽取样本
                idx2 += random.sample(self.M_retrieve['ss'][i], 2)
                idx2 += random.sample(self.M_retrieve['dd'][i], 2)
                idx2 += random.sample(self.M_retrieve['sd'][i], 2)

            #遍历 self.data 的每一个键，如果键对应的值不是列表，那么就将这个值的 idx2 索引对应的元素添加到 samples2 中
            for key in self.data:
                if type(self.data[key]) == list:
                    continue
                else:
                    if type(self.data[key][0]) == np.str_:
                        samples2[key] = self.data[key][idx2].tolist()
                    else:
                        if type(self.data[key]) is not torch.Tensor:
                            samples2[key] = torch.tensor(self.data[key][idx2])
                        else:
                            samples2[key] = self.data[key][idx2]
            return samples2

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        samples = {}
        for key in self.data:
            samples[key] = self.data[key][idx]
        return samples

#batch_size=128
#创建一个数据加载器，用于在训练和测试模型时批量加载数据。
def MOSIDataloader(name, batch_size=None, use_sampler=False, use_similarity=False, simi_return_mono=False, shuffle=True,
                   num_workers=0,
                   prefetch_factor=2,
                   config=default_config):
    if batch_size is None:
        print('batch size not defined')
        return
    dataset = MOSIDataset(name, use_similarity=use_similarity, simi_return_mono=simi_return_mono)
    sampler = None
    drop_last = False

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      sampler=sampler,
                      batch_sampler=None, num_workers=num_workers, collate_fn=None,
                      pin_memory=True, drop_last=drop_last, timeout=0,
                      worker_init_fn=None, prefetch_factor=prefetch_factor,
                      persistent_workers=False)
