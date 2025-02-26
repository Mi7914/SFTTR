import os
import random
import numpy as np
import torch

# 创建目录
def check_dir(path, make_dir=True):
    if not os.path.exists(path):  #
        os.makedirs(path)

#新的日志信息将被添加到文件的末尾。每条日志信息都在新的一行。
def write_log(log, path):
    with open(path, 'a') as f:
        f.writelines(log + '\n')

#设置所有涉及到随机过程的种子，以确保在不同的运行过程中，所有的随机过程的结果都是一致的。
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)


def check_and_save(model, result, check, save_model=True, name=None, parallel=False):
    file_name = 'best_'
    if name is not None:
        file_name = name + '_' + file_name
    for key in check.keys():
        if key == 'MAE' or key == 'Loss':
            #检查了模型的性能指标是否满足保存模型的条件
            if check[key] > result[key]:
                if save_model:
                    if parallel:
                        model.module.save_model(file_name + key)
                    else:
                        model.save_model(file_name + key)
                #更新
                check[key] = result[key]
        else:
            #如果 check 中的值小于 result 中的值，那么就满足条件
            if check[key] < result[key]:
                if save_model:
                    if parallel:
                        model.module.save_model(file_name + key)
                    else:
                        model.save_model(file_name + key)
                #更新
                check[key] = result[key]
    return check


class best_result:
    def __init__(self):
        self.result_save = {'Has0_acc_2': {},
                            'Has0_F1_score': {},
                            'Non0_acc_2': {},
                            'Non0_F1_score': {},
                            'Mult_acc_5': {},
                            'Mult_acc_7': {},
                            'MAE': {},
                            'Corr': {},
                            'Loss': {}}
        self.result_check = {'Has0_acc_2': 0,
                             'Has0_F1_score': 0,
                             'Non0_acc_2': 0,
                             'Non0_F1_score': 0,
                             'Mult_acc_5': 0,
                             'Mult_acc_7': 0,
                             'MAE': 100,
                             'Corr': 0,
                             'Loss': 10000}

    def check(self, result, result_test):
        for key in result.keys():
            if key == 'MAE' or key == 'Loss':
                if self.result_check[key] > result[key]:
                    self.result_check[key] = result[key]
                    self.result_save[key] = result_test
            else:
                if self.result_check[key] < result[key]:
                    self.result_check[key] = result[key]
                    self.result_save[key] = result_test
    #打印结果
    def print(self):
        for key in self.result_save.keys():
            print('==========================')
            print('Best test based on %s' % key)
            for key2 in self.result_save[key].keys():
                print('\t%s: %s' % (key2, self.result_save[key][key2]))
