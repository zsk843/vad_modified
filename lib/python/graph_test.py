import tensorflow as tf

import utils as utils
import numpy as np
import os, sys
import shutil
import wave

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

def bdnn_prediction(batch_size, logits, threshold=0.6, w=19, u=9):
    bdnn_batch_size = batch_size + 2*w
    result = np.zeros((bdnn_batch_size, 1))
    indx = np.arange(bdnn_batch_size) + 1
    indx = indx.reshape((bdnn_batch_size, 1))
    indx = utils.bdnn_transform(indx, w, u)
    indx = indx[w:(bdnn_batch_size-w), :]
    indx_list = np.arange(w, bdnn_batch_size - w)

    for i in indx_list:
        indx_temp = np.where((indx-1) == i)
        pred = logits[indx_temp]
        pred = np.sum(pred)/pred.shape[0]
        result[i] = pred + np.random.rand(1)*1e-4

    result = np.trim_zeros(result)
    soft_result = np.float32(result)
    result = result >= threshold

    return result.astype(int), soft_result


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph


def do_test(fname_model, test_file_dir, norm_dir, prj_dir, is_default, model_type):

    eval_input_dir = test_file_dir


    graph = load_graph(fname_model)

    w = 19
    u = 9
    # [print(n.name) for n in graph.as_graph_def().node]
    # for op in graph.get_operations():
    #     print(op.name)

    final_softout = []
    final_label = []

    if model_type == 0:  # acam
        import data_reader_bDNN_v2 as dr

        print(prj_dir + '/configure/ACAM')
        sys.path.insert(0, os.path.abspath(prj_dir + 'ACAM'))

        # import config as cg
        #
        # if is_default:
        #     w = 19
        #     u = 9
        #     valid_batch_size = 4096
        # else:
        #     w = cg.w
        #     u = cg.u
        #     valid_batch_size = cg.batch_size

        valid_data_set = dr.DataReader(eval_input_dir+"/train/feature_mrcg", eval_input_dir+"/train/label", norm_dir, w=w, u=u, name="eval")
        node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
        node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
        node_keep_probability = graph.get_tensor_by_name('prefix/model_1/keep_probabilty:0')

        node_logits = graph.get_tensor_by_name('prefix/model_1/logits:0')
        node_raw_labels = graph.get_tensor_by_name('prefix/model_1/raw_labels:0')

        acc_sum = 0
        file_num = 0

        acc_sum = 0
        tp_sum = 0
        tn_sum = 0
        fp_sum = 0
        fn_sum = 0
        frame_num = 0
        
        while True:

            valid_inputs, valid_labels = valid_data_set.next_batch(7)

            feed_dict = {node_inputs: valid_inputs, node_labels: valid_labels,
                         node_keep_probability: 1}
            if valid_data_set.eof_checker():
                final_softout = np.reshape(np.asarray(final_softout), [-1, 1])
                final_label = np.reshape(np.asarray(final_label), [-1, 1])
                valid_data_set.reader_initialize()
                # print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
                break

            with tf.Session(graph=graph) as sess:
                logits, raw_labels = sess.run([node_logits, node_raw_labels], feed_dict=feed_dict)

            raw_labels = raw_labels.reshape((-1, 1))
            soft_pred = bdnn_prediction(len(raw_labels), logits, threshold=0.6, w=w, u=u)[0]
            eql = (raw_labels == soft_pred)

            acc = np.mean(eql)
            acc_sum += acc
            file_num+=1

            frame_s = len(raw_labels)
            frame_num += len(raw_labels)
            tn = 0
            tp = 0
            fp = 0
            fn = 0

            for i in range(len(soft_pred)):
                if soft_pred[i] == 0 and raw_labels[i] == 0:
                    tn += 1
                elif soft_pred[i] == 0 and raw_labels[i] == 1:
                    fn += 1
                elif soft_pred[i] == 1 and raw_labels[i] == 0:
                    fp += 1
                elif soft_pred[i] == 1 and raw_labels[i] == 1:
                    tp += 1

            final_softout.append(soft_pred)
            final_label.append(raw_labels)
            tn_sum+=tn
            tp_sum+=tp
            fn_sum+=fn
            fp_sum+=fp

            # if acc <= 0.7:
            #     file_name =valid_data_set.get_cur_file_name().split('/')[-1]
            #     obj_name = file_name.split('.')[0]
            #     wav_path = "/mnt/E_DRIVE/Lipr_with_label/train/low"
            #     shutil.copy("/mnt/E_DRIVE/Lipr_with_label/train/" + obj_name + '.wav', wav_path)
            #     np.save(os.path.join("/mnt/E_DRIVE/Lipr_with_label/train/low", obj_name + '.label.npy'),
            #             original_label(soft_pred, "/mnt/E_DRIVE/Lipr_with_label/train/" + obj_name + '.wav'))

            print(" train_accuracy=%4.4f" % (acc * 100))
            # print("path is " + valid_data_set.get_cur_file_name())
            print("true_positive: %f, false positive: %f, true negative: %f, false negative: %f" % (
                tp / frame_s, fp / frame_s, tn / frame_s, fn / frame_s))

            # print("accuracy: %f, true_positive: %f, false positive: %f, true negative: %f, false negative: %f" % (
            # acc, tp / frame_s, fp / frame_s, tn / frame_s, fn / frame_s))
        print("total accuracy: "+str(acc_sum/file_num))
        print("total: true_positive: %f, false positive: %f, true negative: %f, false negative: %f" % (
        tp_sum / frame_num, fp_sum / frame_num, tn_sum / frame_num, fn_sum / frame_num))


        return final_softout[:, :], final_label[:, :]

    if model_type == 1:  # bdnn
        import data_reader_bDNN_v2 as dr
        print(prj_dir + '/configure/bDNN')
        sys.path.insert(0, os.path.abspath(prj_dir + 'bDNN'))

        import config as cg

        if is_default:
            w = 19
            u = 9
            valid_batch_size = 4096
        else:
            w = cg.w
            u = cg.u
            valid_batch_size = cg.batch_size

        valid_data_set = dr.DataReader(eval_input_dir, eval_output_dir, norm_dir, w=w, u=u, name="eval")  # training data reader initialization
        node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
        node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
        node_keep_probability = graph.get_tensor_by_name('prefix/model_1/keep_probabilty:0')

        node_logits = graph.get_tensor_by_name('prefix/model_1/logits:0')

        while True:
            valid_inputs, valid_labels = valid_data_set.next_batch(valid_batch_size)
            feed_dict = {node_inputs: valid_inputs, node_labels: valid_labels,
                         node_keep_probability: 1}

            if valid_data_set.eof_checker():
                final_softout = np.reshape(np.asarray(final_softout), [-1, 1])
                final_label = np.reshape(np.asarray(final_label), [-1, 1])
                valid_data_set.reader_initialize()
                # print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
                break

            with tf.Session(graph=graph) as sess:
                logits, labels = sess.run([node_logits, node_labels], feed_dict=feed_dict)

            soft_pred = bdnn_prediction(valid_batch_size, logits, threshold=0.6, w=w, u=u)[1]

            raw_indx = int(np.floor(labels.shape[1] / 2))
            raw_labels = labels[:, raw_indx]

            raw_labels = raw_labels.reshape((-1, 1))

            final_softout.append(soft_pred)
            final_label.append(raw_labels)

        return final_softout[0:data_len, :], final_label[0:data_len, :]

    if model_type == 2:  # dnn
        import data_reader_DNN_v2 as dnn_dr
        print(prj_dir + '/configure/DNN')
        sys.path.insert(0, os.path.abspath(prj_dir + 'DNN'))

        import config as cg

        if is_default:
            w = 19
            u = 9
            valid_batch_size = 4096
        else:
            w = cg.w
            u = cg.u
            valid_batch_size = cg.batch_size

        valid_data_set = dnn_dr.DataReader(eval_input_dir, eval_output_dir, norm_dir, w=w, u=u, name="eval")
        node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
        node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
        node_keep_probability = graph.get_tensor_by_name('prefix/model_1/keep_probabilty:0')

        node_softpred = graph.get_tensor_by_name('prefix/model_1/soft_pred:0')
        node_raw_labels = graph.get_tensor_by_name('prefix/model_1/raw_labels:0')
        while True:

            valid_inputs, valid_labels = valid_data_set.next_batch(valid_batch_size)

            one_hot_labels = valid_labels.reshape((-1, 1))
            one_hot_labels = utils.dense_to_one_hot(one_hot_labels, num_classes=2)
            feed_dict = {node_inputs: valid_inputs, node_labels: one_hot_labels,
                         node_keep_probability: 1}
            if valid_data_set.eof_checker():
                final_softout = np.reshape(np.asarray(final_softout), [-1, 1])
                final_label = np.reshape(np.asarray(final_label), [-1, 1])
                valid_data_set.reader_initialize()
                # print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
                break
            with tf.Session(graph=graph) as sess:
                soft_pred, raw_labels = sess.run([node_softpred, node_raw_labels], feed_dict=feed_dict)
            raw_labels = raw_labels.reshape((-1, 1))

            final_softout.append(soft_pred)
            final_label.append(raw_labels)

        return final_softout[0:data_len, :], final_label[0:data_len, :]

    if model_type == 3:  # lstm

        import data_reader_RNN as rnn_dr

        print(prj_dir + '/configure/LSTM')
        sys.path.insert(0, os.path.abspath(prj_dir + 'LSTM'))

        import config as cg

        if is_default:
            target_delay = 5
            seq_size = 20
            batch_num = 200
            valid_batch_size = seq_size * batch_num
        else:
            target_delay = cg.target_delay
            seq_size = cg.seq_len

            valid_batch_size = seq_size

        valid_data_set = rnn_dr.DataReader(eval_input_dir+"/train/feature_mrcg", eval_input_dir+'/train/label', norm_dir, target_delay=target_delay,
                                           name="eval")
        node_inputs = graph.get_tensor_by_name('prefix/model_1/inputs:0')
        node_labels = graph.get_tensor_by_name('prefix/model_1/labels:0')
        node_keep_probability = graph.get_tensor_by_name('prefix/model_1/keep_probabilty:0')

        node_softpred = graph.get_tensor_by_name('prefix/model_1/soft_pred:0')
        node_raw_labels = graph.get_tensor_by_name('prefix/model_1/raw_labels:0')

        while True:

            valid_inputs, valid_labels = valid_data_set.next_batch(valid_batch_size)

            one_hot_labels = valid_labels.reshape((-1, 1))
            one_hot_labels = utils.dense_to_one_hot(one_hot_labels, num_classes=2)
            feed_dict = {node_inputs: valid_inputs, node_labels: one_hot_labels,
                         node_keep_probability: 1}

            if valid_data_set.eof_checker():
                # final_softout = np.reshape(np.asarray(final_softout), [-1, 1])
                # final_label = np.reshape(np.asarray(final_label), [-1, 1])
                valid_data_set.reader_initialize()
                break

            with tf.Session(graph=graph) as sess:
                soft_pred, raw_labels = sess.run([node_softpred, node_raw_labels], feed_dict=feed_dict)
            raw_labels = raw_labels.reshape((-1, 1))

            final_softout.append(soft_pred)
            final_label.append(raw_labels)
            # if valid_data_set.eof_checker():
            #     final_softout = np.reshape(np.asarray(final_softout), [-1, 1])
            #     final_label = np.reshape(np.asarray(final_label), [-1, 1])
            #     valid_data_set.reader_initialize()
            #     # print('Valid data reader was initialized!')  # initialize eof flag & num_file & start index
            #     break

        return final_softout, final_label


def original_label(data, wav_path):
    f = wave.open(wav_path,'r')
    num_frames = f.getnframes()
    splitted_size = num_frames // np.shape(data)[0]
    label = np.zeros(num_frames)
    for i in range(np.shape(data)[0]):
        if data[i] == 1:
            label[i*splitted_size:(i+1)*splitted_size] = 1
    return label


if __name__ == '__main__':
    graph = load_graph('/home/sbie/storage3/github/VAD_Toolkit/VAD/logs/frozen_model.pb')
    print('aa')
    # # Let's allow the user to pass the filename as an argument
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    # parser.add_argument("--test_file_dir", default=0, type=str, help="test_file_dir")
    # parser.add_argument("--prj_dir", default=0, type=str, help="prj_dir")
    # parser.add_argument("--data_len", default=0, type=int, help="data_len")
    # parser.add_argument("--valid_batch_size", default=0, type=int, help="valid_batch_size")
    # parser.add_argument("-m", default=0, type=int, help="model type")
    #
    # args = parser.parse_args()
    #
    # # We use our "load_graph" function
    # graph = load_graph(args.frozen_model_filename)
    #
    # # We can verify that we can access the list of operations in the graph
    # for op in graph.get_operations():
    #     print(op.name)
    #     # prefix/Placeholder/inputs_placeholder
    #     # ...
    #     # prefix/Accuracy/predictions
    #
    # # We access the input and output nodes
    # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    # y = graph.get_tensor_by_name('prefix/Accuracy/predictions:0')
    #
    # # We launch a Session
    # with tf.Session(graph=graph) as sess:
    #     # Note: we don't nee to initialize/restore anything
    #     # There is no Variables in this graph, only hardcoded constants
    #     y_out = sess.run(y, feed_dict={
    #         x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
    #     })
    #     # I taught a neural net to recognise when a sum of numbers is bigger than 45
    #     # it should return False in this case
    #     print(y_out) # [[ False ]] Yay, it works!
