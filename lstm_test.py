import tensorflow as tf
import numpy as np
from data_loader.data_utils import *
from os.path import join as pjoin

n = 400
n_his = 6
n_pred = 1

def multi_pre(sess, y_pred, seq, batch_size, n_his, n_pred, step_idx, dynamic_batch=True):

    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(y_pred,
                            feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
            if isinstance(pred, list):
                pred = np.array(pred[0])
            test_seq[:, 0:n_his - 1, :, :] = test_seq[:, 1:n_his, :, :]
            test_seq[:, n_his - 1, :, :] = pred
            step_list.append(pred)
        pred_list.append(step_list)
    #  pred_array -> [n_pred, batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=1)
    return pred_array[step_idx], pred_array.shape[1]


load_path = f'./output/models_{n}'
data_file = 'data_17_call(1).csv'
n_train, n_val, n_test = 55, 0, 7
if n==307:
    n_train, n_val, n_test = 0.6, 0.2, 0.2
PeMS = data_gen(pjoin('data_loader/PeMS-M/PeMS-M', data_file), (n_train, n_val, n_test), n, n_his + n_pred)
data_test = PeMS.get_data('test')
print(f'>> Loading dataset with Mean: {PeMS.mean}, STD: {PeMS.std}')

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(f'output/models_{n}-199.meta')
    saver.restore(sess, tf.train.latest_checkpoint(f'output'))

    graph = tf.get_default_graph()
    x1 = graph.get_tensor_by_name('X: 0')
    pred_out = graph.get_tensor_by_name('pred:0')

    pred_list = []
    for i in gen_batch(data_test, len(data_test), dynamic_batch=True):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his + 1, :, :])
        test_seq = np.reshape(test_seq,[-1, n_his+1, n])
        step_list = []
        for j in range(n_pred):
            pred = sess.run(pred_out,
                            feed_dict={x1: test_seq})

            # the last step
            pred = pred[:, -1, :]
            test_seq[:, 0:n_his - 1, :] = test_seq[:, 1:n_his, :]
            test_seq[:, n_his - 1, :] = pred
            step_list.append(pred)
        step_list = np.array(step_list)
        # [time_step, batch_size, n] -> [batch_size, time_step, n]
        step_list = np.transpose(step_list, (1, 0, 2))
        pred_list.append(step_list)

    pred_list = np.array(pred_list)
    pred_list = np.reshape(pred_list, [-1, n_pred, n])

    y_true = data_test[:, n_his:, :, :]
    y_true = np.squeeze(y_true)
    pred_list = np.squeeze(pred_list)

    mae = np.mean(np.abs(y_true - pred_list)) * PeMS.std
    rmse = np.sqrt(np.mean(((y_true - pred_list) * PeMS.std) ** 2))
    print(mae, rmse)
    # save_file = f'./dataset_pre/PEMS-{n}/'
    # for i in range(n_pred):
    #     file_name1 = f'true/time_step{i + 1}.csv'
    #     file_name2 = f'pre/time_step{i + 1}.csv'
    #     np.savetxt(pjoin(save_file, file_name1), pred_list[:, i], delimiter=",")
    #     np.savetxt(pjoin(save_file, file_name2), y_true[:, i], delimiter=",")