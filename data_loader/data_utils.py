# @Time     : Jan. 10, 2019 15:26
# @Author   : Veritas YIN
# @FileName : data_utils.py
# @Version  : 1.0
# @IDE      : PyCharm
# @Github   : https://github.com/VeritasYin/Project_Orion

from utils.math_utils import z_score
from os.path import join as pjoin
import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])

    def z_inverse(self, type):
        return self.__data[type] * self.std + self.mean


def seq_gen(data_seq, n_sequence, n_split):
    '''
    '''
    len_s = len(data_seq)

    tmp_seq = []
    i = 0
    while i+n_sequence < len_s:
        sta = i
        end = i+n_sequence
        df_x = data_seq[sta:end]#具体的小切片（12,3）12步3个等待预测的值
        tmp_seq.append(df_x)
        i += n_split
    return np.array(tmp_seq)


def data_gen(file_path, data_config, n_sequence, n_split):

    n_train, n_test = data_config
    # generate training, validation and test data
    print(n_test)

    df_tarin = []
    for path in n_train:
        data_seq = pd.read_csv(pjoin(file_path, f'{path}.csv')).values
        df_tarin.append(data_seq[:, -3:])#对训练数据进行切片，原始为-3，下面代码与此处一致
    df_tarin = np.concatenate(df_tarin)#将数据集拼接一起训练，下面的测试集就不需要这一步
    df_test = []
    for path in n_test:
        data_seq = pd.read_csv(pjoin(file_path, f'{path}.csv')).values
        df_test.append(data_seq[:, -3:])#对训练数据进行切片
    # df_test = np.concatenate(df_test)
    # print(df_test.shape)
    print('>>>>>train shape',df_tarin.shape)

    seq_train = seq_gen(df_tarin, n_sequence, n_split)

    seq_test = []
    for df_test_x in df_test:
        seq_test.append(seq_gen(df_test_x, n_sequence, 1))
    print(np.mean(seq_train[:5]))#输出前5个完整时间序列（5*（12,3）），暂时没有明白作用
    # # x_stats: dict, the stats for the train dataset, including the value of mean and standard deviation.
    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}
    print(seq_train.shape)
    # x_train, x_val, x_test: np.array, [sample_size, n_frame, n_route, channel_size].
    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])#此处用作归一化
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    x_data = {'train': x_train, 'test': x_test}
    dataset = Dataset(x_data, x_stats)

    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    :param inputs: np.ndarray, [len_seq, n_frame, n_route, C_0], standard sequence units.
    :param batch_size: int, the size of batch.
    :param dynamic_batch: bool, whether changes the batch size in the last batch if its length is less than the default.
    :param shuffle: bool, whether shuffle the batches.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
