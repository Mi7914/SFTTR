from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
from model.projector import FeatureProjector
from model.decoder.classifier import BaseClassifier
import torch
from torch import nn
import config as default_config


#文本编码器，使用bert
class TextEncoder(nn.Module):
    #初始化
    def __init__(self, name=None, fea_size=None, proj_fea_dim=None, drop_out=None, with_projector=True,
                 config=default_config):
        super(TextEncoder, self).__init__()
        self.name = name

        if fea_size is None:
            fea_size = config.MOSI.downStream.text_fea_dim# 768
        if proj_fea_dim is None:
            proj_fea_dim = config.MOSI.downStream.proj_fea_dim#256
        if drop_out is None:
            drop_out = config.MOSI.downStream.text_drop_out#0.5

        # 根据配置中的 USEROBERTA 的值，选择加载 “roberta-base” 或 “bert-base-uncased” 的预训练模型，
        # 并将它们的分词器和模型赋给 self.tokenizer 和 self.extractor
        if config.USEROBERTA:
            self.tokenizer = BertTokenizer.from_pretrained("roberta-base")
            self.extractor = BertModel.from_pretrained("roberta-base")
        else:
            self.tokenizer = BertTokenizer.from_pretrained('/data/tm/ConFEDE-Graph_Main/bert-base-uncased/')
            self.extractor = BertModel.from_pretrained('/data/tm/ConFEDE-Graph_Main/bert-base-uncased/')
        self.with_projector = with_projector
        #如果 with_projector 为 True，那么就创建一个 FeatureProjector 对象，并将它赋给 self.projector
        #设置投影层
        if with_projector:
            self.projector = FeatureProjector(fea_size, proj_fea_dim, drop_out=drop_out,
                                              name='text_projector', config=config)
        self.device = config.DEVICE

    #首先使用分词器对文本进行处理，然后使用模型提取特征，最后如果 self.with_projector 为 True，那么就使用投影器对特征进行处理。
    def forward(self, text, device=None):
        if device is None:
            device = self.device

        x = self.tokenizer(text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(
            device)
        x = self.extractor(**x)['pooler_output']
        # x = self.extractor(**x)['last_hidden_state']
        if self.with_projector:
            x = self.projector(x)
        return x

    #设置训练状态
    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True]
        # 如果 train_module[0] 为 True，那么模型的参数需要计算梯度，否则不需要
        for name, param in self.extractor.named_parameters():
                param.requires_grad = train_module[0]

        #如果 self.with_projector 为 True 并且 train_module[1] 为 True，那么投影器的参数需要计算梯度，否则不需要。
        if self.with_projector:
            for param in self.projector.parameters():
                param.requires_grad = train_module[1]

#对文本进行预训练，包括编码文本、预测标签、计算损失、保存和加载模型以及设置参数是否需要计算梯度等操作
class TextPretrain(nn.Module):
    #初始化
    def __init__(self, name=None, proj_fea_dim=768, drop_out=None, config=default_config):
        super(TextPretrain, self).__init__()
        if drop_out is None:
            drop_out = config.MOSI.downStream.text_drop_out#0.5
        #初始化文本编码器
        self.encoder = TextEncoder(name=name, with_projector=False)  # bert output 768
        #初始化回归分类器
        self.classifier = BaseClassifier(input_size=proj_fea_dim,
                                         hidden_size=[int(proj_fea_dim / 2), int(proj_fea_dim / 4),
                                                      int(proj_fea_dim / 8)],
                                         output_size=1, drop_out=drop_out, name='RegClassifier', )
        self.device = config.DEVICE
        self.criterion = torch.nn.MSELoss()
        self.config=config

    #前向传播
    def forward(self, text, label, return_loss=True, device=None):
        if device is None:
            device = self.device
        #将文本编码为特征向量x
        x = self.encoder(text, device=device)
        #使用回归分类器预测输出
        pred = self.classifier(x)
        #如果要计算损失，则计算MSE损失
        if return_loss:
            loss = self.criterion(pred.squeeze(), label.squeeze())
            return pred, x, loss
        else:
            return pred, x
        
    #保存模型的权重到文件
    def save_model(self, name: object) -> object:
        # save all modules
        encoder_path = self.config.MOSI.path.encoder_path + name + '_text_encoder.ckpt'
        decoder_path = self.config.MOSI.path.encoder_path + name + '_text_decoder.ckpt'
        torch.save(self.encoder.state_dict(), encoder_path)
        torch.save(self.classifier.state_dict(), decoder_path)
        print('model saved at:')
        print(encoder_path)
        print(decoder_path)
    
    #从文件加载模型的权重
    def load_model(self, name, module=None):
        encoder_path = self.config.MOSI.path.encoder_path + name + '_text_encoder.ckpt'
        decoder_path = self.config.MOSI.path.encoder_path + name + '_text_decoder.ckpt'
        print('model loaded from:')
        if module == 'encoder':
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            print(encoder_path)
        if module == 'decoder':
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(decoder_path)
        if module == 'all' or module is None:
            self.encoder.load_state_dict(torch.load(encoder_path, map_location=self.device))
            self.classifier.load_state_dict(torch.load(decoder_path, map_location=self.device))
            print(encoder_path)
            print(decoder_path)

    #设置训练模式
    def set_train(self, train_module=None):
        if train_module is None:
            train_module = [True, True, True]
        self.encoder.set_train(train_module=train_module[0:2])
        #控制在反向传播过程中是否应为 self.classifier 的参数计算梯度
        for param in self.classifier.parameters():
            param.requires_grad = train_module[2]
