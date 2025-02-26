from model.net.constrastive.text_encoder_finetune import TextEncoder
from model.net.constrastive.vision_encoder_finetune import VisionEncoder
from model.net.constrastive.audio_encoder_fintune import AudioEncoder
import torch
import config as default_config
from torch import nn
from model.decoder.classifier import BaseClassifier
from util.metrics import weighted_NTXentLoss
import numpy as np
from util.CrossModal import CrossModel
from util.common import check_dir


# class projector(nn.Module):
#     def __init__(self, input_dim, output_dim, dropout=0.5):
#         super(projector, self).__init__()
#
#         self.fc = nn.Sequential(
#             nn.LayerNorm(input_dim),
#             nn.Linear(input_dim, output_dim),
#             # nn.ReLU(),
#             # nn.Linear(output_dim, output_dim),
#             nn.Tanh(),
#             nn.Dropout(dropout)
#         )
#
#     def forward(self, x):
#         x = self.fc(x)
#         return x
class common_feature_extractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.3):
        super(common_feature_extractor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.fc(x)
        return x

class private_feature_extractor(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        super(private_feature_extractor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        x = self.fc(x)
        return x

class TVA_fusion(nn.Module):
    def __init__(self, name=None, encoder_fea_dim=None, drop_out=None, config=default_config):
        super(TVA_fusion, self).__init__()
        self.config = config
        self.text_encoder = TextEncoder(name=name, with_projector=False, config=config)
        self.vision_encoder = VisionEncoder(config=config)
        self.audio_encoder = AudioEncoder(config=config)
        if encoder_fea_dim is None:
            encoder_fea_dim = config.MOSEI.downStream.encoder_fea_dim
        if drop_out is None:
            drop_out = config.MOSEI.downStream.text_drop_out

        uni_fea_dim = int(encoder_fea_dim/2)

        self.T_simi_proj = common_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.V_simi_proj = common_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.A_simi_proj = common_feature_extractor(encoder_fea_dim, uni_fea_dim)

        self.T_dissimi_proj = private_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.V_dissimi_proj = private_feature_extractor(encoder_fea_dim, uni_fea_dim)
        self.A_dissimi_proj = private_feature_extractor(encoder_fea_dim, uni_fea_dim)

        # 模型融合层
        self.vat_cross = CrossModel(config)
        self.va_cross = CrossModel(config)
        self.pc_cross = CrossModel(config)

        hidden_size = [uni_fea_dim * 2, uni_fea_dim, int(uni_fea_dim / 2), int(uni_fea_dim / 4),
                       ]

        self.TVA_decoder = BaseClassifier(input_size=uni_fea_dim,
                                          hidden_size=hidden_size,
                                          output_size=1, drop_out=drop_out,
                                          name='TVARegClassifier', )

        self.mono_decoder = BaseClassifier(input_size=uni_fea_dim,
                                           hidden_size=hidden_size[2:],
                                           output_size=1, drop_out=drop_out,
                                           name='TVAMonoRegClassifier', )

        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.model_path = config.MOSEI.path.model_path + str(config.seed) + '/'
        check_dir(self.model_path)

        self.batch_size = config.MOSEI.downStream.TVAExp_fusion.batch_size
        self.heat = config.MOSEI.downStream.const_heat

        self.ntxent_loss = weighted_NTXentLoss(temperature=self.heat)
        self.set_train()

    def forward(self, sample1, sample2, return_loss=True, return_emb=False, device=None):
        if device is None:
            device = self.device

        text1 = sample1['raw_text']
        vision1 = sample1['vision'].clone().detach().to(device).float()
        audio1 = sample1['audio'].clone().detach().to(device).float()
        label1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        label_T1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        label_V1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        label_A1 = sample1['regression_labels'].clone().detach().to(device).float()  # .squeeze()
        key_padding_mask_V1, key_padding_mask_A1 = (sample1['vision_padding_mask'].clone().detach().to(device),
                                                    sample1['audio_padding_mask'].clone().detach().to(device))

        x_t_embed = self.text_encoder(text1, device=device).squeeze()
        x_v_embed = self.vision_encoder(vision1, key_padding_mask=key_padding_mask_V1, device=device).squeeze()
        x_a_embed = self.audio_encoder(audio1, key_padding_mask=key_padding_mask_A1, device=device).squeeze()

        x_t_simi1 = self.T_simi_proj(x_t_embed)
        x_v_simi1 = self.V_simi_proj(x_v_embed)
        x_a_simi1 = self.A_simi_proj(x_a_embed)
        x_t_dissimi1 = self.T_dissimi_proj(x_t_embed)
        x_v_dissimi1 = self.V_dissimi_proj(x_v_embed)
        x_a_dissimi1 = self.A_dissimi_proj(x_a_embed)

        x1_s = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1), dim=-1)
        x1_ds = torch.cat((x_t_dissimi1, x_v_dissimi1, x_a_dissimi1), dim=-1)
        x1_all = torch.cat((x1_s, x1_ds), dim=-1)
        x1_sds = torch.cat((x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1,), dim=0)
        x_sds = x1_sds
        # 将公共特征连接
        common_feature1 = x_t_simi1 + x_v_simi1 + x_a_simi1
        # <========= common and private feature extractor
        # 得到common_mask
        mask1 = torch.ones(x_t_simi1.unsqueeze(dim=1).shape[:2]).clone().detach().to(device)
        common_mask1 = torch.ones(common_feature1.unsqueeze(dim=1).shape[:2]).clone().detach().to(device)
        # 获得最后表示cross_output
        cross_output_1, cross_mask_1 = self._get_cross_output(x_t_dissimi1.unsqueeze(dim=1),
                                                              x_v_dissimi1.unsqueeze(dim=1),
                                                              x_a_dissimi1.unsqueeze(dim=1),
                                                              common_feature1.unsqueeze(dim=1), mask1, mask1,
                                                              mask1, common_mask1)
        label1_sds = torch.cat((label1,label1,label1,label_T1, label_V1, label_A1,), dim=0)
        x = cross_output_1.mean(dim=1)
        label_sds = label1_sds
        label_all = label1.squeeze()
        if sample2 is not None:
            text2 = sample2['raw_text']
            vision2 = sample2['vision'].clone().detach().to(device).float()
            audio2 = sample2['audio'].clone().detach().to(device).float()
            label2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_T2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_V2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            label_A2 = sample2['regression_labels'].clone().detach().to(device).float()  # .squeeze()
            key_padding_mask_V2, key_padding_mask_A2 = (sample2['vision_padding_mask'].clone().detach().to(device),
                                                        sample2['audio_padding_mask'].clone().detach().to(device))

            x_t_embed2 = self.text_encoder(text2, device=device).squeeze()
            x_v_embed2 = self.vision_encoder(vision2, key_padding_mask=key_padding_mask_V2, device=device).squeeze()
            x_a_embed2 = self.audio_encoder(audio2, key_padding_mask=key_padding_mask_A2, device=device).squeeze()

            x_t_simi2 = self.T_simi_proj(x_t_embed2)
            x_v_simi2 = self.V_simi_proj(x_v_embed2)
            x_a_simi2 = self.A_simi_proj(x_a_embed2)
            x_t_dissimi2 = self.T_dissimi_proj(x_t_embed2)
            x_v_dissimi2 = self.V_dissimi_proj(x_v_embed2)
            x_a_dissimi2 = self.A_dissimi_proj(x_a_embed2)

            # x2_s = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2), dim=-1)
            # x2_ds = torch.cat((x_t_dissimi2, x_v_dissimi2, x_a_dissimi2), dim=-1)
            # x2_all = torch.cat((x2_s, x2_ds), dim=-1)
            # x2_sds = torch.cat((x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2, x_a_dissimi2,), dim=0)
            #
            # # 将公共特征连接
            # common_feature2 = x_t_simi2 + x_v_simi2 + x_a_simi2
            # # <========= common and private feature extractor
            # # 得到common_mask
            # mask2 = torch.ones(x_t_simi2.unsqueeze(dim=1).shape[:2]).clone().detach().to(device)
            # common_mask2 = torch.ones(common_feature2.unsqueeze(dim=1).shape[:2]).clone().detach().to(device)
            #
            # # 获得最后表示cross_output
            # cross_output_2, cross_mask_2 = self._get_cross_output(x_t_dissimi2.unsqueeze(dim=1),
            #                                                       x_v_dissimi2.unsqueeze(dim=1),
            #                                                       x_a_dissimi2.unsqueeze(dim=1),
            #                                                       common_feature2.unsqueeze(dim=1), mask2,
            #                                                       mask2,
            #                                                       mask2, common_mask2)
            #
            #
            #
            # label2_sds = torch.cat((label2,label2,label2,label_T2, label_V2, label_A2,), dim=0)
            # x = torch.cat((cross_output_1, cross_output_2), dim=0).mean(dim=1)
            # label_all = torch.cat((label1.squeeze(), label2.squeeze()), dim=0)
            # x_sds = torch.cat((x1_sds, x2_sds), dim=0)
            # label_sds = torch.cat((label1_sds, label2_sds), dim=0)

        if return_loss:
            pred = self.TVA_decoder(x)
            pred_mono = self.mono_decoder(x_sds)
            sup_const_loss = 0
            # sds_loss = 0
            if sample2 is not None:
                # [Ts,T1s,T2s,T3s,T4s,T5s,T6s,V1s,V2s,V3s,....]
                t1, p, t2, n = torch.tensor([0, 0, 7, 7, 14, 14,
                                             0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                                            device=device), \
                               torch.tensor([1, 2, 8, 9, 15, 16,
                                             7, 14, 8, 15, 9, 16, 10, 17, 11, 18, 12, 19, 13, 20],
                                            device=device), \
                               torch.tensor([0, 0, 0, 0, 7, 7, 7, 7, 14, 14, 14, 14,
                                             0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6],
                                            device=device), \
                               torch.tensor([3, 4, 5, 6, 10, 11, 12, 13, 17, 18, 19, 20,
                                             21, 28, 35, 22, 29, 36, 23, 30, 37, 24, 31, 38, 25, 32, 39, 26, 33, 40, 27,
                                             34, 41], device=device)

                indices_tuple = (t1, p, t2, n)
                pre_sample_label = torch.tensor([0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 1, 2, 3, 4,
                                                 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, 5, 5, 5, 6, 7, 8, 9, ])
                for i in range(len(x1_all)):
                    pre_sample_x = []
                    for fea1, fea2 in zip([x_t_simi1, x_v_simi1, x_a_simi1, x_t_dissimi1, x_v_dissimi1, x_a_dissimi1, ],
                                          [x_t_simi2, x_v_simi2, x_a_simi2, x_t_dissimi2, x_v_dissimi2,
                                           x_a_dissimi2, ]):
                        pre_sample_x.append(torch.cat((fea1[i].unsqueeze(0), fea2[6 * i:6 * (i + 1)]), dim=0))

                    sup_const_loss += self.ntxent_loss(torch.cat(pre_sample_x, dim=0), pre_sample_label,
                                                       indices_tuple=indices_tuple)

                sup_const_loss /= len(x1_all)

            pred_loss = self.criterion(pred.squeeze(), label_all)

            loss = pred_loss + 0.1 * sup_const_loss
            if return_emb:
                return pred, x1_all, loss, pred_loss, sup_const_loss
            else:
                return pred, (x_t_embed, x_v_embed, x_a_embed), loss, pred_loss, sup_const_loss
        else:
            if return_emb:
                return x1_all
            else:
                return (x_t_embed, x_v_embed, x_a_embed)

    def _get_cross_output(self, sequence_output, visual_output, audio_output, common_feature, attention_mask,
                          visual_mask, audio_mask, common_mask):
        # =============> visual audio fusion
        #连接视频和音频输出
        va_concat_features = torch.cat((audio_output, visual_output), dim=1)
        #构造视频音频mask
        # audio_mask = torch.tensor(audio_mask).unsqueeze(0)
        # visual_mask = torch.tensor(visual_mask).unsqueeze(0)

        #连接视频音频mask
        va_concat_mask = torch.cat((visual_mask, visual_mask), dim=1)
        # va_concat_mask = torch.cat((audio_mask, visual_mask), dim=1)
        #构造文本、视频、音频type_mask
        text_type_ = torch.zeros_like(attention_mask)#0
        video_type_ = torch.ones_like(visual_mask)#1
        audio_type_ = torch.zeros_like(audio_mask)#0
        audio_type_new = torch.zeros_like(visual_mask)#0

        #连接视频音频type的mask
        va_concat_type = torch.cat((audio_type_new, video_type_), dim=1)#(0,1)
        #获得视频音频经过CrossModel的输出

        # print(va_concat_features.size()) [32, 768]
        # print(va_concat_type.size())
        # print(va_concat_mask.size())
        va_cross_layers = self.va_cross(va_concat_features, va_concat_type, va_concat_mask)
        #获得输出

        va_cross_output = va_cross_layers[-1]
        # <============= visual audio fusion

        # =============> VisualAudio and text fusion


        #从维度为1融合三种模态融合特征
        vat_concat_features = torch.cat((sequence_output, va_cross_output), dim=1)
        #连接三种模态的mask
        vat_concat_mask = torch.cat((attention_mask, va_concat_mask), dim=1)
        # 获得三种模态的type_mask
        va_type_ = torch.ones_like(va_concat_mask)#1
        vat_concat_type = torch.cat((text_type_, va_type_), dim=1)#(0,1)
        #获取三种模态融合之后的输出
        vat_cross_layers = self.vat_cross(vat_concat_features, vat_concat_type, vat_concat_mask)
        vat_cross_output = vat_cross_layers[-1]
        # <============= VisualAudio and text fusion

        # =============> private common fusion
        #连接相似特征和不相似特征
        pc_concate_features = torch.cat((vat_cross_output, common_feature), dim=1)
        specific_type = torch.zeros_like(vat_concat_mask)#0
        common_type = torch.ones_like(common_mask)#1
        pc_concate_type = torch.cat((specific_type, common_type), dim=1)#（0，1）
        pc_concat_mask = torch.cat((vat_concat_mask, common_mask), dim=1)
        #获取融合后的特征
        pc_cross_layers = self.pc_cross(pc_concate_features, pc_concate_type, pc_concat_mask)
        pc_cross_output = pc_cross_layers[-1]
        # <============= private common fusion

        return pc_cross_output, pc_concat_mask

    def save_model(self, name):
        # save all modules
        mode_path = self.model_path + 'TVA_fusion' + '_model.ckpt'

        print('model saved at:')
        print(mode_path)
        torch.save(self.state_dict(), mode_path)

    def load_model(self, name, load_pretrain=False):
        if load_pretrain:
            text_encoder_path = self.config.MOSEI.path.encoder_path + name + '_text_encoder.ckpt'
            vision_encoder_path = self.config.MOSEI.path.encoder_path + name + '_vision_encoder.ckpt'
            audio_encoder_path = self.config.MOSEI.path.encoder_path +name + '_audio_encoder.ckpt'

            print('model loaded from:')
            print(text_encoder_path)
            print(vision_encoder_path)
            print(audio_encoder_path)
            self.text_encoder.load_state_dict(torch.load(text_encoder_path, map_location=self.device))
            # self.text_encoder.tokenizer.from_pretrained(self.config.SIMS.path.bert_en,do_lower_case=True)
            # self.text_encoder.extractor.from_pretrained(self.config.SIMS.path.bert_en)
            self.vision_encoder.load_state_dict(torch.load(vision_encoder_path, map_location=self.device))
            self.audio_encoder.load_state_dict(torch.load(audio_encoder_path, map_location=self.device))

        else:
            mode_path = self.model_path + 'TVA_fusion' + '_model.ckpt'

            print('model loaded from:')
            print(mode_path)
            self.load_state_dict(torch.load(mode_path, map_location=self.device))

    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [False, False, True, True]

        for param in self.parameters():
            param.requires_grad = train_module[3]
        self.text_encoder.set_train(train_module=train_module[0:2])
        self.vision_encoder.set_train(train_module=train_module[2])
        self.audio_encoder.set_train(train_module=train_module[2])
