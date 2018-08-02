import sys
sys.path.insert(0, './lib/python')
import VAD_Proposed as Vp
import VAD_DNN as Vd
import VAD_bDNN as Vb
import VAD_LSTM_2 as Vl
import scipy.io as sio
import graph_test as graph_test
import os, getopt
import glob

from time import time

# norm_dir = "./norm_data"
# data_dir = "./sample_data"
# ckpt_name = '/model9918and41.ckpt-2'
# model_dir = "./saved_model"
# valid_batch_size = 4134

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'hm:', ["data_dir=", "prj_dir="])
    except getopt.GetoptError as err:
        print(str(err))
        sys.exit(1)

    if len(opts) != 3:
        print("arguments are not enough.")
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            sys.exit(0)
        elif opt == '-m':
            mode = int(arg)
        elif opt == '--data_dir':
            data_dir = str(arg)
            norm_dir = data_dir+'/train/feature_mrcg/global_normalization'
        elif opt == '--prj_dir':
            prj_dir = str(arg)
            model_dir = prj_dir+'/saved_model'
    
   
    is_default = False

    if mode == 0:

        if is_default:
            graph_list = sorted(glob.glob(model_dir + '/backup/backup_pb/frozen_model_ACAM.pb'))
            norm_dir = model_dir + '/backup/backup_norm'
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)
        else:
            graph_list = sorted(glob.glob(model_dir + '/graph/ACAM/*.pb'))
            print(graph_list)
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)

    elif mode == 1:

        print(prj_dir+'/configure/bDNN')
        sys.path.insert(0, os.path.abspath(prj_dir+'/configure/bDNN'))

        import config as cg
        if is_default:
            graph_list = sorted(glob.glob(model_dir + '/backup/backup_pb/frozen_model_bDNN.pb'))
            norm_dir = model_dir + '/backup/backup_norm'
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)
        else:
            graph_list = sorted(glob.glob(model_dir + '/graph/bDNN/*.pb'))
            print(graph_list)
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)

    elif mode == 2:

        start_time = time()
        print(prj_dir + '/configure/DNN')
        sys.path.insert(0, os.path.abspath(prj_dir + '/configure/DNN'))

        import config as cg

        if is_default:
            graph_list = sorted(glob.glob(model_dir + '/backup/backup_pb/frozen_model_DNN.pb'))
            norm_dir = model_dir + '/backup/backup_norm'
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)
        else:
            graph_list = sorted(glob.glob(model_dir + '/graph/DNN/*.pb'))
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)

        end_time = time()

        time_taken = end_time - start_time
        print(time_taken)

    elif mode == 3:

        print(prj_dir + '/configure/LSTM')
        sys.path.insert(0, os.path.abspath(prj_dir + '/configure/LSTM'))

        import config as cg

        if is_default:
            graph_list = sorted(glob.glob(model_dir + '/backup/backup_pb/frozen_model_LSTM.pb')) 
            norm_dir = model_dir + '/backup/backup_norm'
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)
        else:
            graph_list = sorted(glob.glob(model_dir + '/graph/LSTM/*.pb'))
            print(graph_list)
            pred, label = graph_test.do_test(graph_list[-1], data_dir, norm_dir, prj_dir, is_default, mode)

        for p in pred:
            p[p <= 0] = 0
            p[p > 0] = 1

    
    sio.savemat('../../result/pred.mat', {'pred': pred})
    sio.savemat('../../result/label.mat', {'label': label})
    print("done")
