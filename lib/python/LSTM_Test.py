import tensorflow as tf
import numpy as np
import utils as utils
import re
import data_reader_RNN as dr
import sys, os
import matplotlib.pyplot as plt
from tensorflow.contrib import rnn
from sklearn import metrics
import time
from functools import reduce
from operator import mul
import graph_save as gs
import shutil
import wave


FLAGS = tf.flags.FLAGS
SEED = 1
tf.set_random_seed(SEED)

file_dir = "/home/sbie/storage2/VAD_Database/SE_TIMIT_MRCG_0328"
input_dir = file_dir
output_dir = file_dir + "/Labels"
valid_file_dir = "/home/sbie/storage2/VAD_Database/NX_TIMIT_MRCG_big"

# valid_file_dir = "/media/jskim/F440795840792312/database/jskim/NX_TIMIT_MRCG_big"

norm_dir = input_dir

logs_dir = "./saved_model"
initial_logs_dir = "/home/sbie/github/VAD_Project_test/VAD_LSTM/logs_LSTM"
ckpt_name = "/LSTM"

reset = True  # remove all existed logs and initialize log directories
device = '/gpu:3'

mode = 'test'
if mode is 'test':
    reset = False

if reset:

    os.popen('rm -rf ' + logs_dir + '/*')
    os.popen('mkdir ' + logs_dir + '/train')
    os.popen('mkdir ' + logs_dir + '/valid')

summary_list = ["cost", "accuracy_SNR_-5", "accuracy_SNR_0", "accuracy_SNR_5", "accuracy_SNR_10",
                "accuracy_across_all_SNRs"]

learning_rate = 0.0001
eval_num_batches = 2e4
SMALL_NUM = 1e-4
max_epoch = int(1e5)
dropout_rate = 0.5

decay = 0.9  # batch normalization decay factor
target_delay = 5  # target_delay default = 19
u = 9  # u default = 9
eval_th = 0.6
th = 0.5
lstm_cell_size = 128
num_layers = 3

model_config = {'target_delay': target_delay, "u": u}
seq_size = 20
batch_num = 200
batch_size = batch_num*seq_size # batch_size = 32
valid_batch_size = batch_size

# assert (target_delay-1) % u == 0, "target_delay-1 must be divisible by u"

width = 768
num_features = 256  # MRCG feature
bdnn_winlen = (((target_delay-1) / u) * 2) + 3

# bdnn_inputsize = int(bdnn_winlen * num_features)
bdnn_inputsize = num_features
bdnn_outputsize = 2#int(bdnn_winlen)
initLr = 0.01
scope_name = 'RNN_scope'
eval_type = 2


def train_config(c_train_dir, c_valid_dir, c_logs_dir, c_seq_size, c_batch_num, c_max_epoch, c_mode):

    global file_dir
    global input_dir
    global output_dir
    global valid_file_dir
    global norm_dir
    global initial_logs_dir
    global logs_dir
    global ckpt_name
    global batch_size
    global valid_batch_size
    global mode
    global max_epoch
    global seq_size
    global batch_num

    file_dir = c_train_dir
    valid_file_dir = c_valid_dir
    input_dir = file_dir
    output_dir = file_dir + "/Labels"

    norm_dir = file_dir
    initial_logs_dir = logs_dir = c_logs_dir
    # batch_size = valid_batch_size = c_batch_size_eval + 2 * target_delay
    seq_size = c_seq_size
    batch_num = c_batch_num

    batch_size = valid_batch_size = c_seq_size * c_batch_num

    max_epoch = c_max_epoch
    mode = c_mode


def test_config(c_test_dir, c_norm_dir, c_initial_logs_dir, c_seq_size, c_batch_num, c_data_len):

    global test_file_dir
    global norm_dir
    global initial_logs_dir
    global ckpt_name
    global valid_batch_size
    global batch_size
    global data_len
    global batch_num
    global seq_size
    global batch_num

    test_file_dir = c_test_dir
    norm_dir = c_norm_dir
    initial_logs_dir = c_initial_logs_dir

    seq_size = c_seq_size
    batch_num = c_batch_num
    valid_batch_size = batch_size = c_seq_size * c_batch_num
    data_len = c_data_len
    batch_num = batch_size/seq_size


def affine_transform(x, output_dim, name=None):
    """
    affine transformation Wx+b
    assumes x.shape = (batch_size, num_features)
    """

    w = tf.get_variable(name + "_w", [x.get_shape()[1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=0.02))
    b = tf.get_variable(name + "_b", [output_dim], initializer=tf.constant_initializer(0.0))

    return tf.matmul(x, w) + b


def rnn_in(inputs, seq_size, delay):

    seq_size = int(seq_size)
    delay = int(delay)
    temp1 = tf.reshape(inputs, [-1, num_features])
    temp2 = tf.reshape(temp1[0:-delay, :], [-1, seq_size, num_features])
    temp3 = temp2[1:, 0:delay, :]
    temp4 = tf.reshape(temp1[-delay:, :], [-1, delay, num_features])
    temp5 = tf.concat([temp3, temp4], 0)
    return tf.concat([temp2, temp5], 1)


def summary_generation(eval_file_dir):

    summary_dic = {}

    noise_list = os.listdir(eval_file_dir)
    noise_list = sorted(noise_list)
    summary_dic["summary_ph"] = summary_ph = tf.placeholder(dtype=tf.float32)

    for name in noise_list:

        with tf.variable_scope(name):
            for summary_name in summary_list:
                    summary_dic[name+"_"+summary_name] = tf.summary.scalar(summary_name, summary_ph)

    with tf.variable_scope("Averaged_Results"):

        summary_dic["cost_across_all_noise_types"] = tf.summary.scalar("cost_across_all_noise_types", summary_ph)
        summary_dic["accuracy_across_all_noise_types"]\
            = tf.summary.scalar("accuracy_across_all_noise_types", summary_ph)
        summary_dic["variance_across_all_noise_types"]\
            = tf.summary.scalar("variance_across_all_noise_types", summary_ph)
        summary_dic["AUC_across_all_noise_types"]\
            = tf.summary.scalar("AUC_across_all_noise_types", summary_ph)
    return summary_dic


def full_evaluation(m_eval, sess_eval, batch_size_eval, eval_file_dir, summary_writer, summary_dic, itr):

    mean_cost = []
    mean_accuracy = []
    mean_auc = []
    mean_time = []

    print("-------- Performance for each of noise types --------")

    noise_list = os.listdir(eval_file_dir)
    noise_list = sorted(noise_list)

    summary_ph = summary_dic["summary_ph"]

    for i in range(len(noise_list)):
        print("full time evaluation, now loading : %d",i)
        noise_name = '/' + noise_list[i]
        eval_input_dir = eval_file_dir + noise_name
        eval_output_dir = eval_file_dir + noise_name + '/Labels'
        ##########################################


        eval_calc_dir = eval_file_dir + noise_name + '/test_result' # for Final layer information saving



        ##########################################
        eval_data_set = dr.DataReader(eval_input_dir, eval_output_dir, norm_dir, target_delay=target_delay, u=u, name="eval")

        eval_cost, eval_accuracy, eval_list, eval_auc, auc_list, eval_time = evaluation(m_eval, eval_data_set, sess_eval,  batch_size_eval, noise_list[i], save_dir = eval_calc_dir)

        print("--noise type : " + noise_list[i])
        print("cost: %.3f, accuracy across all SNRs: %.3f, auc across all SNRs: %.3f " % (eval_cost, eval_accuracy, eval_auc))

        print('accuracy wrt SNR:')

        print('SNR_-5 : %.3f, SNR_0 : %.3f, SNR_5 : %.3f, SNR_10 : %.3f' % (eval_list[0], eval_list[1],
                                                                            eval_list[2], eval_list[3]))

        print('AUC wrt SNR:')

        print('SNR_-5 : %.3f, SNR_0 : %.3f, SNR_5 : %.3f, SNR_10 : %.3f' % (auc_list[0], auc_list[1],
                                                                            auc_list[2], auc_list[3]))
        eval_summary_list = [eval_cost] + eval_list + [eval_accuracy] + [eval_auc]

        for j, summary_name in enumerate(summary_list):
            summary_str = sess_eval.run(summary_dic[noise_list[i]+"_"+summary_name],
                                        feed_dict={summary_ph: eval_summary_list[j]})
            summary_writer.add_summary(summary_str, itr)

        mean_cost.append(eval_cost)
        mean_accuracy.append(eval_accuracy)
        mean_auc.append(eval_auc)
        mean_time.append(eval_time)

    mean_cost = np.mean(np.asarray(mean_cost))
    var_accuracy = np.var(np.asarray(mean_accuracy))
    mean_accuracy = np.mean(np.asarray(mean_accuracy))
    mean_auc = np.mean(np.asarray(mean_auc))
    mean_time = np.mean(np.asarray(mean_time))

    summary_writer.add_summary(sess_eval.run(summary_dic["cost_across_all_noise_types"],
                                             feed_dict={summary_ph: mean_cost}), itr)
    summary_writer.add_summary(sess_eval.run(summary_dic["accuracy_across_all_noise_types"],
                                             feed_dict={summary_ph: mean_accuracy}), itr)
    summary_writer.add_summary(sess_eval.run(summary_dic["variance_across_all_noise_types"],
                                             feed_dict={summary_ph: var_accuracy}), itr)
    summary_writer.add_summary(sess_eval.run(summary_dic["AUC_across_all_noise_types"],
                                             feed_dict={summary_ph: mean_auc}), itr)

    print("-------- Performance across all of noise types --------")
    print("cost : %.3f" % mean_cost)
    print("******* averaged accuracy across all noise_types : %.3f *******" % mean_accuracy)
    print("******* variance of accuracies across all noise_types : %6.6f *******" % var_accuracy)
    print("******* variance of AUC across all noise_types : %6.6f *******" % mean_auc)
    print("******* mean time : %6.6f *******" % mean_time)
    return mean_auc


def evaluation(m_valid, valid_data_set, sess, eval_batch_size, noise_name, num_batches=eval_num_batches,
               save_dir = None):
    # num_samples = valid_data_set.num_samples
    # num_batches = num_samples / batch_size
    avg_valid_cost = 0.
    avg_valid_accuracy = 0.
    avg_valid_time = 0.
    # AUC = 0.
    itr_sum = 0.
    file_num_before = -1
    accuracy_list = [0 for i in range(valid_data_set._file_len)]
    cost_list = [0 for i in range(valid_data_set._file_len)]
    auc_list = [0 for i in range(valid_data_set._file_len)]
    time_list = [0 for i in range(valid_data_set._file_len)]
    itr_file = 0
    channel_values = []
    save_calc_dir = ''
    valid_name_before = ''
    # plt.figure()
    while True:

        # valid_name_before = save_dir + 'test'+ [file_num_before] + '.txt'
        valid_inputs, valid_labels = valid_data_set.next_batch(eval_batch_size)

        if valid_data_set.file_change_checker():

            accuracy_list[itr_file] = avg_valid_accuracy / itr_sum
            cost_list[itr_file] = avg_valid_cost / itr_sum
            auc_list[itr_file] = utils.plot_ROC2(channel_values,file_num_before, noise_name)
            time_list[itr_file] = avg_valid_time / ((itr_sum*batch_size)/16000)
            avg_valid_accuracy = 0.
            avg_valid_cost = 0.
            avg_valid_time = 0.
            channel_values = []
            # avg_valid_auc = 0.
            itr_sum = 0
            itr_file += 1
            valid_data_set.file_change_initialize()

        if valid_data_set.eof_checker():
            #######
            # AUC = utils.plot_ROC2(valid_name_before, save_calc_dir,file_num_before, noise_name)
            # f_eval.close()
            #######
            valid_data_set.reader_initialize()
            print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
            break

        # if eval_batch_size * itr_file > 5000:
        #     f_eval.close()
        #     break

        one_hot_vlabels = valid_labels.reshape((-1, 1))
        one_hot_vlabels = dense_to_one_hot(one_hot_vlabels, num_classes=2)
        feed_dict = {m_valid.inputs: valid_inputs, m_valid.labels: one_hot_vlabels,
                     m_valid.keep_probability: 1}
        start_time = time.time()
        logits_val = sess.run(m_valid.logits, feed_dict=feed_dict)
        check_time = time.time()-start_time
        valid_cost, valid_accuracy = sess.run([m_valid.cost, m_valid.accuracy], feed_dict=feed_dict)
        # print(valid_labels.shape)
        # print(logits_val)

        save_val = np.concatenate([logits_val,valid_labels], axis = 1)
        ###############################################

        file_num = valid_data_set._num_file
        channel_values.append(save_val)

        avg_valid_cost += valid_cost
        avg_valid_accuracy += valid_accuracy
        avg_valid_time += check_time
        # avg_valid_auc += valid_auc
        itr_sum += 1

        ############################################
        file_num_before = valid_data_set._num_file
        ############################################

    total_avg_valid_cost = np.asscalar(np.mean(np.asarray(cost_list)))
    total_avg_valid_accuracy = np.asscalar(np.mean(np.asarray(accuracy_list)))
    total_avg_valid_auc = np.asscalar(np.mean(np.asarray(auc_list)))
    total_avg_valid_time = np.asscalar(np.mean(np.asarray(time_list)))
    return total_avg_valid_cost, total_avg_valid_accuracy, accuracy_list, total_avg_valid_auc, auc_list, total_avg_valid_time


def dense_to_one_hot(labels_dense, num_classes=2):

    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[(index_offset + labels_dense.ravel()).astype(int)] = 1
    return labels_one_hot.astype(np.float32)


class Model(object):

    def __init__(self, is_training=True):

        self.keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
        self.inputs = inputs = tf.placeholder(tf.float32, shape=[None, bdnn_inputsize],name="inputs")
        self.labels = labels = tf.placeholder(tf.float32, shape=[None, bdnn_outputsize], name="labels")

        lrDecayRate = 0.96
        lrDecayFreq = 20000

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(learning_rate, self.global_step, lrDecayFreq, lrDecayRate, staircase=True)

        self.logits = logits = self.inference(inputs, self.keep_probability, is_training=is_training)  # (batch_size, bdnn_outputsize)

        pred = tf.argmax(logits, axis=1, name="prediction")
        self.pre = pred
        self.softpred = tf.identity(pred, name="soft_pred")

        pred = tf.cast(pred, tf.int32)
        truth = tf.cast(labels[:, 1], tf.int32)

        self.raw_labels = tf.identity(truth, name="raw_labels")

        log_one = logits[:, 1]

        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(pred, truth), tf.float32))
        self.cost = cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits = logits))

        trainable_var = tf.trainable_variables()
        self.train_op = self.train(cost, trainable_var)
        self.seq_len = seq_size

    def train(self, loss_val, var_list):

        optimizer = tf.train.AdamOptimizer(self.lr)
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)

        return optimizer.apply_gradients(grads, global_step=self.global_step)

    def inference(self, inputs, keep_prob, is_training=True, reuse=None):
        # initialization
        # c1_out = affine_transform(inputs, num_hidden_1, name="hidden_1")
        # inputs_shape = inputs.get_shape().as_list()
        with tf.variable_scope(scope_name):
            # print(inputs.get_shape().as_list())
            in_rnn = rnn_in(inputs, seq_size, target_delay)

            # in_rnn = tf.reshape(inputs,[-1, seq_size+target_delay, num_features])
            stacked_rnn = []
            for iiLyr in range(num_layers):
                stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(num_units=lstm_cell_size, state_is_tuple=True))
            MultiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(cells=stacked_rnn, state_is_tuple=True)

            outputs, _state = tf.nn.dynamic_rnn(MultiLyr_cell, in_rnn, time_major=False, dtype=tf.float32)
            outputs = tf.reshape(outputs[:, 0:seq_size, :], [-1, lstm_cell_size])

            outputs = tf.nn.dropout(outputs, keep_prob=keep_prob)

            # # h1_out = affine_transform(inputs, num_hidden_1, name="hidden_1")
            # lh1_out = utils.batch_norm_affine_transform(outputs, num_hidden_1, name="lhidden_1", decay=decay,
            #                                             is_training=is_training)
            # lh1_out = tf.nn.relu(lh1_out)
            # lh1_out = tf.nn.dropout(lh1_out, keep_prob=keep_prob)

            logits = affine_transform(outputs, bdnn_outputsize, name="output1")
            # logits = tf.sigmoid(logits)
            logits = tf.reshape(logits, [-1, int(bdnn_outputsize)])

        return logits


def main(save_dir, prj_dir=None, model=None, mode=None, dev="/gpu:2"):

    device = dev
    os.environ["CUDA_VISIBLE_DEVICES"] = device[-1]

    import path_setting as ps
    set_path = ps.PathSetting(prj_dir, model, save_dir)
    logs_dir = initial_logs_dir = set_path.logs_dir
    input_dir = set_path.input_dir
    output_dir = set_path.output_dir
    norm_dir = set_path.norm_dir
    valid_file_dir = set_path.valid_file_dir

    sys.path.insert(0, prj_dir + '/configure/LSTM')
    import config as cg

    global seq_size, batch_num

    seq_size = cg.seq_len
    batch_num = cg.num_batches

    global learning_rate, dropout_rate, max_epoch, batch_size, valid_batch_size
    learning_rate = cg.lr
    dropout_rate = cg.dropout_rate
    max_epoch = cg.max_epoch
    batch_size = valid_batch_size = batch_num * seq_size

    global target_delay
    target_delay = cg.target_delay

    global lstm_cell_size, num_layers
    lstm_cell_size = cg.cell_size
    num_layers = cg.num_layers

    print("Graph initialization...")
    with tf.device(device):
        with tf.variable_scope("model", reuse=None):
            m_train = Model(is_training=True)
        with tf.variable_scope("model", reuse=True):
            m_valid = Model(is_training=False)

    print("Done")

    print("Setting up Saver...")
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(logs_dir)

    print("Done")

    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    if ckpt and ckpt.model_checkpoint_path:  # model restore
        print("Model restored...")

        if mode is 'train':
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            saver.restore(sess, initial_logs_dir+ckpt_name)
            saver.save(sess, logs_dir + "/model_LSTM.ckpt", 0)  # model save

        print("Done")

    train_data_set = dr.DataReader(input_dir, output_dir, norm_dir, target_delay=target_delay, u=u, name="train")

    if mode is 'train':
        file_len = train_data_set.get_file_len()
        acc_sum = 0
        tp_sum = 0
        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        frame_num = 0
        for itr in range(file_len):

            train_inputs, train_labels = train_data_set.next_batch(seq_size)

            one_hot_labels = train_labels.reshape((-1, 1))
            one_hot_labels = dense_to_one_hot(one_hot_labels, num_classes=2)
            feed_dict = {m_train.inputs: train_inputs, m_train.labels: one_hot_labels,
                         m_train.keep_probability: dropout_rate}

            train_accuracy, train_pre, raw_label = sess.run([m_train.accuracy, m_train.pre, m_train.raw_labels], feed_dict=feed_dict)


            frame_s = len(train_pre)
            frame_num += len(train_pre)
            tn = 0
            tp = 0
            fp = 0
            fn = 0

            for i in range(len(train_pre)):
                if train_pre[i] == 0 and raw_label[i] == 0:
                    tn += 1
                elif train_pre[i] == 0 and raw_label[i] == 1:
                    fn += 1
                elif train_pre[i] == 1 and raw_label[i] == 0:
                    fp += 1
                elif train_pre[i] == 1 and raw_label[i] == 1:
                    tp += 1


            acc_sum += train_accuracy
            tn_sum += tn
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn

            # if train_accuracy <= 0.7:
            #     file_name = train_data_set.get_cur_file_name().split('/')[-1]
            #     obj_name = file_name.split('.')[0]
            #     wav_path = "/mnt/E_DRIVE/TIMIT_echo_noisy/train/low"
            #     shutil.copy("/mnt/E_DRIVE/TIMIT_echo_noisy/train/"+obj_name+'.wav', wav_path)
            #     np.save(os.path.join("/mnt/E_DRIVE/TIMIT_echo_noisy/train/low", obj_name+'.label.npy'), original_label(train_pre,"/mnt/E_DRIVE/TIMIT_echo_noisy/train/"+obj_name+'.wav'))

            print("Step: %d/%d, train_accuracy=%4.4f" % (file_len, itr, train_accuracy * 100))
            # print("path is "+train_data_set.get_cur_file_name())
            print("true_positive: %f, false positive: %f, true negative: %f, false negative: %f" % (
            tp / frame_s, fp / frame_s, tn / frame_s, fn / frame_s))

        # valid_file_reader = dr.DataReader(valid_file_dir+'/feature_mrcg', valid_file_dir+'/label', norm_dir, target_delay = target_delay, u=u, name="train")
        # valid_len = valid_file_reader.get_file_len()
        # valid_accuracy, valid_cost = utils.do_validation(m_valid, sess, valid_file_dir, norm_dir, type='LSTM')
        # total_acc = (valid_accuracy*valid_len+acc_sum)/(valid_len+file_len)

        total_acc = acc_sum/file_len
        print("valid_accuracy=%4.4f" % (total_acc * 100))
        print("total: true_positive: %f, false positive: %f, true negative: %f, false negative: %f"%(tp_sum/frame_num, fp_sum/frame_num, tn_sum/frame_num, fn_sum/frame_num))


def original_label(data, wav_path):
    f = wave.open(wav_path,'r')
    num_frames = f.getnframes()
    splitted_size = num_frames // np.shape(data)[0]
    label = np.zeros(num_frames)
    for i in range(np.shape(data)[0]):
        if data[i] == 1:
            label[i*splitted_size:(i+1)*splitted_size] = 1
    return label


def get_num_params():
    num_params = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        num_params += reduce(mul, [dim.value for dim in shape], 1)
    return num_params


if __name__ == "__main__":

    if len(sys.argv) == 4:
        gpu_no = sys.argv[1]
        prj_dir = sys.argv[2]
        data_dir = sys.argv[3]
        main(data_dir, prj_dir, 'LSTM', 'train', dev='/gpu:'+gpu_no)
