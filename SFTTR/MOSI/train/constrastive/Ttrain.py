import torch
import config as default_config
from model.net.constrastive.text_encoder_finetune import TextPretrain
from dataloader.MOSI import MOSIDataloader
from tqdm import tqdm
import transformers as trans
from util.metrics import Metrics
from util.common import write_log, check_and_save
import datetime


#文本预训练
def Ttrain(exp_type=None, load_model=None, check=None, config=default_config):
    print('---------------TextPretrain---------------')
    if check is None:
        check = {'Non0_F1_score': 0, 'MAE': 100, 'Loss': 10000}
    else:
        check = check.copy()

    log_path = config.LOGPATH + "MOSI_TextPretrain." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    #config.LOGPATH :'log/'
    #获取了当前的日期和时间，并将其格式化为 ‘年-月-日-小时分钟’ 的形式

    #获取训练以及测试数据
    train_data = MOSIDataloader('train', batch_size=config.MOSI.downStream.textPretrain.batch_size)
    valid_data = MOSIDataloader('valid', shuffle=False, batch_size=config.MOSI.downStream.textPretrain.batch_size,
                                num_workers=0)
    metrics = Metrics()#评价指标

    #加载模型
    #对文本进行预训练，包括编码文本、预测标签、计算损失、保存和加载模型以及设置参数是否需要计算梯度等操作
    model = TextPretrain(config=config)

    device = config.DEVICE
    batch_size = config.MOSI.downStream.textPretrain.batch_size
    lr = config.MOSI.downStream.textPretrain.lr#1e-4
    total_epoch = config.MOSI.downStream.textPretrain.epoch#200
    decay = config.MOSI.downStream.textPretrain.decay#1e-3
    num_warm_up = config.MOSI.downStream.textPretrain.num_warm_up#5

    #创建了一个 AdamW 优化器，它是一种常用的优化算法，可以自适应地调整学习率。params=model.parameters() 指定了需要优化的参数，这些参数通常是模型的权重和偏置。
    #lr=lr 设置了学习率，它决定了参数更新的步长。weight_decay=decay 设置了权重衰减，它是一种正则化技术，可以防止模型过拟合。
    optimizer = trans.optimization.AdamW(params=model.parameters(), lr=lr, weight_decay=decay)

    #创建一个学习率调度器，它可以根据训练的进度动态地调整学习率。
    #这个调度器的策略是：在初始的 num_warmup_steps 步内，学习率从0线性增加到预设的学习率；然后在剩余的步骤中，学习率从预设的学习率线性降低到0。
    scheduler = trans.optimization.get_linear_schedule_with_warmup(optimizer,
                                                                   num_warmup_steps=int(
                                                                       num_warm_up * (len(train_data))),
                                                                   num_training_steps=total_epoch * len(train_data), )
    model.to(device)
    #从文件加载模型的权重
    if load_model is not None:
        model.load_model(load_model, module='encoder')
    model.train()
    loss = 0

    train_all_epoch = int(total_epoch / 3)
    for epoch in range(1, total_epoch + 1):
        #根据 epoch 是否小于 train_all_epoch 来设置模型的训练模式。
        if epoch < train_all_epoch:
            model.set_train([False, True, True])
        else:
            model.set_train([True, True, True])
        #创建了一个进度条 bar，它用于显示训练数据的处理进度。
        bar = tqdm(train_data, disable=False)
        for index, sample in enumerate(bar):
            #设置了进度条的描述，包括当前的轮次和损失。
            bar.set_description("Epoch:%d|Loss:[%s]|" % (epoch, loss))
            #将优化器中的所有梯度设置为零。
            optimizer.zero_grad()
            #首先从样本中获取文本和标签，然后将文本和标签输入到模型中，得到预测值、特征和损失。
            text = sample['raw_text']
            label = sample['regression_labels'].clone().detach().to(device)
            pred, fea, loss = model(text, label.float().squeeze(), return_loss=True)
            loss.backward()

            optimizer.step()
            scheduler.step()

        #评估模型在验证数据上的性能，并得到结果和损失。
        result, result_loss = eval(model, metrics, valid_data, device, config)

        #构造了一个日志字符串，它包含了当前轮次、各种评估指标的结果和损失。
        log = 'visionPretarin_TrainAcc\n\tEpoch:%d\n\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
                  epoch, result['Has0_acc_2'], result['Has0_F1_score'],
                  result['Non0_acc_2'], result['Non0_F1_score'], result['Mult_acc_5'],
                  result['Mult_acc_7'], result['MAE'], result['Corr'], result_loss)
        print(log)
        write_log(log, path=log_path)
        #检查 epoch 是否大于 train_all_epoch，如果是，那么就检查并保存模型。
        if epoch > train_all_epoch:
            check = check_and_save(model, result, check)
    print(check)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

#用于评估神经网络模型的性能。
def eval(model, metrics=None, eval_data=None, device=None, config=default_config):
    if device is None: device = config.DEVICE
    #获取数据集
    if eval_data is None: eval_data = MOSIDataloader('test', shuffle=False, num_workers=0,
                                                     batch_size=config.MOSI.downStream.textPretrain.batch_size)
    if metrics is None: metrics = Metrics()#评价指标

    #将模型设置为评价模式
    model.eval()
    #禁用了梯度计算，这可以减少内存使用并加速计算，但是在训练过程中是不需要的。
    with torch.no_grad():
        pred = []
        truth = []
        loss = 0
        bar = tqdm(eval_data, disable=True)
        for index, sample in enumerate(bar):
            text = sample['raw_text']
            label = sample['regression_labels'].clone().detach().to(device)
            _pred, fea, _loss = model(text, label.float().squeeze(), return_loss=True)
            #预测
            pred.append(_pred.view(-1))
            truth.append(label)
            #计算预测值和真实值之间的损失。
            loss += _loss.item() * config.MOSI.downStream.textPretrain.batch_size

        pred = torch.cat(pred).to(torch.device('cpu'), ).squeeze()
        truth = torch.cat(truth).to(torch.device('cpu'))

        #计算了预测值和真实值的评价指标，并将平均损失添加到评价结果中。
        eval_results = metrics.eval_mosei_regression(truth, pred)
        eval_results['Loss'] = loss / len(eval_data)
    #将模型设置为训练模式。
    model.train()
    return eval_results, loss / len(eval_data)

#对预训练的文本模型进行测试，包括加载模型、进行评价、记录日志等操作。
def Ttest(check_list=None, config=default_config):
    if check_list is None: check_list = ['Has0_acc_2', 'Has0_F1_score', 'Mult_acc_7']
    #检查 check_list 是否为列表，如果不是，那么将 check_list 转换为列表。
    if not isinstance(check_list, list): check_list = [check_list]
    log_path = config.LOGPATH + "MOSI_TextPretrain_Test." + datetime.datetime.now().strftime(
        '%Y-%m-%d-%H%M') + '.log'
    #加载模型
    model = TextPretrain(config=config)
    device = config.DEVICE
    model.to(device)
    #首先初始化了一个字典 check，然后遍历 check_list 中的每一个指标，
    # 对每一个指标加载模型并进行评价，最后将评价结果保存到 check 中
    check = {}
    for metric in check_list:
        print('Result for best ' + metric)
        model.load_model(name='best_' + metric)
        result, loss = eval(model=model, device=device, config=config)
        check[metric] = result[metric]

        log = '\tHas0_acc_2:%s\n\tHas0_F1_score:%s\n\tNon0_acc_2"%s\n\t' \
              'Non0_F1_score:%s\n\tMult_acc_5:%s\n\tMult_acc_7:%s\n\tMAE:%s\n\tCorr:%s\n\tLoss:%s\n' \
              '------------------------------------------' % (
                  result['Has0_acc_2'], result['Has0_F1_score'],
                  result['Non0_acc_2'], result['Non0_F1_score'], result['Mult_acc_5'],
                  result['Mult_acc_7'], result['MAE'], result['Corr'], loss)

        print(log)
        write_log(metric + '\n' + log, log_path)
