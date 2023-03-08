import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.python.ops.rnn import dynamic_rnn
import time
from data_loader.data_utils import *
from os.path import join as pjoin

in_uint = 128
rnn_unit = 128
epoch = 150
lr = 0.001
batch_size =50
n_sequence = 12
n_split = 1
c_out = 3


def lstm(x, rnn_unit):
    # 定义LSTM网络
    b = tf.shape(x)[0]
    # tf.nn.rnn_cell.BasicLSTMCell
    # tf.nn.rnn_cell.BasicRNNCell
    # tf.nn.rnn_cell.GRUCell
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_unit)
    init_state = cell.zero_state(b, dtype=tf.float32)

    output_rnn, final_state = dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)
    print('>>>>>>>>>>',final_state.h.shape)
    # print(final_state.shape)
    # output_rnn = tf.reshape(final_state.h, [-1,rnn_unit])
    output_rnn = final_state.h
    output_rnn = tf.layers.dense(output_rnn, units=c_out)

    return output_rnn


def train(inputs, n_sequence):
    # 训练函数部分
    X = tf.placeholder(dtype=tf.float32, shape=[None, n_sequence, c_out], name='X')

    x = X[:, :n_sequence-1, :]
    y = X[:, n_sequence-1, :]

    pred = lstm(x, rnn_unit)

    print(pred.shape)
    # pred_out = tf.reshape(pred, [-1, n_his, n], name='pred')
    train_loss = tf.reduce_mean(tf.abs(y - pred))

    train_op = tf.train.AdamOptimizer(lr).minimize(train_loss)
    saver = tf.train.Saver(max_to_keep=10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            start_time = time.time()
            print(f'>>>>>epoch {i} start training....')
            total_loss = 0
            for j, x_batch in enumerate(
                    gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                # x_batch = np.reshape(x_batch,[-1, n_his+n_pred, n])
                loss, _ = sess.run([train_loss, train_op], feed_dict={X: x_batch})
                total_loss += loss
            print(f'>>>>>epoch {i}:', total_loss)
        # 打印输出的格式
        test = inputs.get_data('test')
        for i, df_test in enumerate(test):
            y_pre = []
            for j, x_batch in enumerate(gen_batch(df_test, batch_size, dynamic_batch=True, shuffle=False)):
                test_pre = sess.run([pred], feed_dict={X: x_batch})
                y_pre.append(np.array(test_pre)[0])
            y_pre = np.concatenate(y_pre)
            y_true = df_test[:, -1, :]
            y_pre = y_pre * inputs.std + inputs.mean
            y_true = y_true*inputs.std + inputs.mean
            np.savetxt(f'pre/pre_{i}.csv', y_pre, delimiter=',', fmt='%.4f')
            np.savetxt(f'pre/true_{i}.csv', y_true, delimiter=',', fmt='%.4f')
#             输出生成的预测文件和测试文件

file_name = [[1, 2 ], [1, 2, 3, 4, 5]]
# 文件存放命名和训练、测试集的选择，测试时只需要更改文件名称即可
file_path = 'data'
df = data_gen(file_path, file_name, n_sequence, n_split)
print('>> Loading dataset ')
print("出现此代码说明另外一个文件运行成功")
#
train(df, n_sequence)