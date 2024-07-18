import torch
import config as default_config
from model.net.constrastive.TVA_fusion import TVA_fusion
from dataloader.MOSI import MOSIDataloader
from tqdm import tqdm
import transformers as trans
from util.metrics import Metrics
from util.common import write_log, check_and_save
import datetime


#更新模型的相似性矩阵
def update_matrix(data, model, config=default_config):
    with torch.no_grad():
        #评估模式，接下来的前向传播不会改变模型的参数。
        model.eval()
        #加载训练数据
        train_data = MOSIDataloader('train', batch_size=32, use_similarity=False, simi_return_mono=False, shuffle=False,
                                    use_sampler=False)
        device = config.DEVICE
        print('Collecting New embeddings')
        T, V, A = [], [], []
        bar = tqdm(train_data, disable=True)
        for index, sample in enumerate(bar):
            #前向传播，得到前途向量
            _T, _V, _A = model(sample, None, return_loss=False)
            T.append(_T.detach())
            V.append(_V.detach())
            A.append(_A.detach())
        T = torch.cat(T, dim=0).to(torch.device('cpu')).squeeze()
        V = torch.cat(V, dim=0).to(torch.device('cpu')).squeeze()
        A = torch.cat(A, dim=0).to(torch.device('cpu')).squeeze()
        print('Updating Similarity Matrix')
        #更新数据集的相似性矩阵。
        data.dataset.update_matrix(T, V, A)
        #将模型设置回训练模式。
        model.train()
    return

#训练一个融合模型，包括设置模型的训练模式、加载模型、处理训练数据、计算损失和梯度、更新参数和学习率、进行评价等操作。
#在训练过程中的特定轮次保存模型。
def TVA_train_fusion(name, load_model=None, check=None, model_type='all', load_pretrain=True,
                     config=default_config):
    print('---------------TVA_EXP_%s---------------' % model_type)
    if check is None:
        check = {'Loss': 10000, 'MAE': 100}
    else:
        check = check.copy()
    log_path = config.LOGPATH + "MOSI_TVA_experiment." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    metrics = Metrics()#评价指标
    # 设置模型的训练模式。
    train_bool = [False, False, True, True, True]
    #模型融合
    model = TVA_fusion(config=config)
    model.to(config.DEVICE)
    #设置模型的训练模式。
    model.set_train(train_bool)
    #设置参数
    device = config.DEVICE
    batch_size = config.MOSI.downStream.TVAExp_fusion.batch_size
    lr = config.MOSI.downStream.TVAExp_fusion.lr
    total_epoch = config.MOSI.downStream.TVAExp_fusion.epoch
    decay = config.MOSI.downStream.TVAExp_fusion.decay
    num_warm_up = config.MOSI.downStream.TVAExp_fusion.num_warm_up
    finetune_epoch = config.MOSI.downStream.TVAExp_fusion.finetune_epoch
    #获取训练数据
    train_data = MOSIDataloader('train', batch_size=batch_size, use_similarity=True, simi_return_mono=False,
                                use_sampler=False)
    #创建了一个参数组列表，它用于设置优化器
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    #创建了一个 AdamW 优化器和一个学习率调度器。
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=lr, amsgrad=False, )
    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        num_warm_up * (len(train_data))),
                                                                   num_training_steps=total_epoch * len(train_data),
                                                                   )

    #检查 load_model 是否为 None，如果不是，那么加载模型。
    if load_model is not None:
        model.load_model(load_model, load_pretrain=load_pretrain)

    loss = 0
    all_loss = 0
    sub_loss = 0
    save_start_epoch = 1
    #在每一个轮次中，它更新了矩阵、设置了模型的训练模式、遍历了训练数据中的每一个样本、计算了损失
    # 进行了反向传播、更新了参数和学习率，最后进行了评价。
    for epoch in range(1, total_epoch + 1):
        if epoch % 2 == 1:
            update_matrix(train_data, model, config)
        model.train()
        if epoch == finetune_epoch:
            train_bool = [True, True, True, True, True]
            model.set_train(train_bool)
        bar = tqdm(train_data, disable=False)
        for index, sample1, in enumerate(bar):
            try:
                bar.set_description("Epoch:%d|All_loss:%s|Loss:%s|SDS_loss:%s" % (
                    epoch, all_loss.item(), loss.item(), sub_loss.item()))
            except:
                bar.set_description(
                    "Epoch:%d|All_loss:%s|Loss:%s|SDS_loss:%s" % (epoch, all_loss, loss, sub_loss))

            optimizer.zero_grad()

            idx = sample1['index']
            sample2 = train_data.dataset.sample(idx)

            pred, fea, all_loss, loss, sub_loss = model(sample1, sample2, return_loss=True, device=device)

            all_loss.backward()

            optimizer.step()
            scheduler.step()

        print("EVAL valid")
        result, result_loss = eval(model, metrics, 'valid', device, config)
        log = 'TVA_%s_ValidAcc\n\tEpoch:%d\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (model_type,
                                                              epoch, result['Has0_acc_2'],
                                                              result['Has0_F1_score'],
                                                              result['Non0_acc_2'], result['Non0_F1_score'],
                                                              result['Mult_acc_5'],
                                                              result['Mult_acc_7'], result['MAE'],
                                                              result['Corr'],
                                                              result_loss)
        print(log)
        write_log(log, path=log_path)

        #检查模型的性能，并在满足特定条件时保存模型。
        if epoch > save_start_epoch:
            check = check_and_save(model, result, check, save_model=True, name=name)

#测试模型性能
def eval(model, metrics=None, eval_data=None, device=None, config=default_config):
    with torch.no_grad():
        model.eval()
        if device is None: device = config.DEVICE
        if eval_data is None:
            eval_data = MOSIDataloader('test', shuffle=False, num_workers=0,
                                       batch_size=config.MOSI.downStream.TVAExp_fusion.batch_size)
        else:
            eval_data = MOSIDataloader(eval_data, shuffle=False, num_workers=0,
                                       batch_size=config.MOSI.downStream.TVAExp_fusion.batch_size)
        if metrics is None: metrics = Metrics()
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            label = sample['regression_labels'].clone().detach().to(device).float()
            _pred, fea, _all_loss, _loss, _ = model(sample, None, return_loss=True)
            pred.append(_pred.view(-1))
            truth.append(label)
            loss += _loss.item() * config.MOSI.downStream.TVAExp_fusion.batch_size
        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / len(eval_data)
        model.train()
    return eval_results, loss / len(eval_data)


#测试一个融合模型，包括加载模型、进行评价、记录日志等操作。
def TVA_test_fusion(name, check_list=None, model_type='all', config=default_config):
    if check_list is None: check_list = ['Has0_F1_score', 'MAE']
    #检查 check_list 是否为列表，如果不是，那么将 check_list 转换为列表。
    if not isinstance(check_list, list): check_list = [check_list]
    seed = config.seed
    #路径
    log_path = config.LOGPATH + "MOSI_TVA_experiment_Test." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '_seed_' + str(seed) + '.log'
    #模型融合
    model = TVA_fusion(config=config)

    device = config.DEVICE
    model.to(device)
    check = {}
    result = None
    print('Evaluating model:' + model_type)
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name=name + '_best_' + metric)
        result, loss = eval(model=model, device=device, config=config)
        check[metric] = {}
        for key in result.keys():
            check[metric][key] = result[key]

        log = 'TVA_%s_TestAcc\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (model_type,
                                                              result['Has0_acc_2'], result['Has0_F1_score'],
                                                              result['Non0_acc_2'], result['Non0_F1_score'],
                                                              result['Mult_acc_5'],
                                                              result['Mult_acc_7'], result['MAE'], result['Corr'], loss)

        print(log)
        write_log(metric + '\n' + log, log_path)

    return check
