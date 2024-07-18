import torch
import os
from util.common import check_dir

seed = [1,12]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
LOGPATH = 'log/'
check_dir(LOGPATH)

USEROBERTA = False


class MOSI:
    class path:
        bert_en = '/data/tm/C+T/ConFEDE/data/MOSI/ckpt/bert_en/'
        raw_data_path = '/data/tm/ConFEDE-Graph_Main/data/MOSI/Processed/unaligned_50.pkl'
        model_path = '/data/tm/1.27/ckpt/encoder/'
        encoder_path = '/data/tm/1.27/ckpt/fea_encoder/'
        if USEROBERTA:
            model_path = model_path + '/roberta/'
        else:
            model_path = model_path + '/bert/'
        check_dir(model_path)
        result_path = 'result/'
        check_dir(result_path)

    class downStream:
        # follow below performance
        metric = 'MAE'
        load_metric = 'best_' + metric
        check_list = [metric]

        # select which model to save
        check = {metric: 10000 if metric == 'Loss' or metric == 'MAE' else 0}

        # parameters
        use_reg = True
        proj_fea_dim = 256
        encoder_fea_dim = 768
        text_fea_dim = 768
        # vision_fea_dim = 35
        vision_fea_dim = 20
        video_seq_len = 500
        audio_fea_dim = 5
        audio_seq_len = 375
        text_drop_out = 0.5
        vision_drop_out = 0.5
        audio_drop_out = 0.5
        vision_nhead = 8
        audio_nhead = 8
        vision_dim_feedforward = vision_fea_dim
        audio_dim_feedforward = audio_fea_dim
        vision_tf_num_layers = 2
        audio_tf_num_layers = 2
        num_attention_heads = 8 #12
        num_hidden_layers = 2 #maybe 12
        hidden_act = "gelu"#maybe
        hidden_size = 384 #maybe 768
        attention_probs_dropout_prob = 0.1 #0.3
        hidden_dropout_prob = 0.1#0.3
        intermediate_size = 256 #3072
        max_position_embeddings = 240  # 60,240,512
        type_vocab_size = 2

        sds_heat = 0.5
        const_heat = 0.5

        class textPretrain:
            batch_size = 128
            lr = 1e-4
            epoch = 200
            decay = 1e-3
            num_warm_up = 5

        class visionPretrain:
            batch_size = 128
            lr = 1e-4
            epoch = 100
            decay = 1e-3
            num_warm_up = 5

        class audioPretrain:
            batch_size = 128
            lr = 1e-4
            epoch = 100
            decay = 1e-3
            num_warm_up = 10

        class TVAExp_fusion:
            batch_size = 16
            lr = 1e-4
            epoch = 25
            decay = 1e-3
            num_warm_up = 1
            finetune_epoch = 200