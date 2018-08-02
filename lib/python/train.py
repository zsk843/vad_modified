import sys

sys.path.insert(0, './lib/python')
import VAD_Proposed as Vp
import VAD_DNN as Vd
import VAD_bDNN as Vb
import VAD_LSTM_2 as Vl
import scipy.io as sio
import os, getopt
import time
import graph_save as gs
import path_setting as ps
# norm_dir = "./norm_data"
# data_dir = "./sample_data"
# ckpt_name = '/model9918and41.ckpt-2'
# model_dir = "./saved_model"
# valid_batch_size = 4134
import sys

mode = 3
extract_feat = 0
prj_dir = r"/home/yckj/Documents/jtkim"

if __name__ == '__main__':

    if len(sys.argv) >= 2:
        mode = int(sys.argv[1])

    gpu_no = '3'
    
    if len(sys.argv) >= 3:
        gpu_no = sys.argv[2]

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = gpu_no

    if len(sys.argv) >= 4:
        prj_dir = sys.argv[3]

    data_dir = prj_dir + '/data/mrcg_data/feat'

    if len(sys.argv) >= 5:
        data_dir = sys.argv[4]

    train_data_dir = data_dir + '/train'
    valid_data_dir = data_dir + '/valid'

    save_dir = data_dir
    train_save_dir = save_dir + '/train'
    valid_save_dir = save_dir + '/valid'

    if extract_feat:

        os.system("rm -rf " + save_dir)
        os.system("mkdir " + save_dir)
        os.system("mkdir " + save_dir + '/train')
        os.system("mkdir " + save_dir + '/valid')
        os.system(
            "matlab -r \"try acoustic_feat_ex(\'%s\',\'%s\'); catch; end; quit\"" % (train_data_dir, train_save_dir))
        os.system(
            "matlab -r \"try acoustic_feat_ex(\'%s\',\'%s\'); catch; end; quit\"" % (valid_data_dir, valid_save_dir))

        train_norm_dir = save_dir + '/train/global_normalize_factor.mat'
        test_norm_dir = prj_dir + '/norm_data/global_normalize_factor.mat'

        os.system("cp %s %s" % (train_norm_dir, test_norm_dir))

    if mode == 0:
        set_path = ps.PathSetting(prj_dir, 'ACAM', save_dir)
        logs_dir = set_path.logs_dir

        os.system("rm -rf " + logs_dir + '/train')
        os.system("rm -rf " + logs_dir + '/valid')
        os.system("mkdir " + logs_dir + '/train')
        os.system("mkdir " + logs_dir + '/valid')

        Vp.main(save_dir, prj_dir, 'ACAM', 'train', dev='/gpu:'+gpu_no)

        gs.freeze_graph(prj_dir + '/logs/ACAM', prj_dir + '/saved_model/graph/ACAM', 'model_1/logits,model_1/raw_labels')

    if mode == 1:
        set_path = ps.PathSetting(prj_dir, 'bDNN', save_dir)
        logs_dir = set_path.logs_dir

        os.system("rm -rf " + logs_dir + '/train')
        os.system("rm -rf " + logs_dir + '/valid')
        os.system("mkdir " + logs_dir + '/train')
        os.system("mkdir " + logs_dir + '/valid')

        Vb.main(save_dir, prj_dir, 'bDNN', 'train',dev='/gpu:'+gpu_no)

        gs.freeze_graph(prj_dir + '/logs/bDNN', prj_dir + '/saved_model/graph/bDNN', 'model_1/logits,model_1/labels')

    if mode == 2:
        set_path = ps.PathSetting(prj_dir, 'DNN', save_dir)
        logs_dir = set_path.logs_dir

        os.system("rm -rf " + logs_dir + '/train')
        os.system("rm -rf " + logs_dir + '/valid')
        os.system("mkdir " + logs_dir + '/train')
        os.system("mkdir " + logs_dir + '/valid')

        Vd.main(save_dir, prj_dir, 'DNN', 'train',dev='/gpu:'+gpu_no)

        gs.freeze_graph(prj_dir + '/logs/DNN', prj_dir + '/saved_model/graph/DNN', 'model_1/soft_pred,model_1/raw_labels')

    if mode == 3:
        set_path = ps.PathSetting(prj_dir, 'LSTM', save_dir)
        logs_dir = set_path.logs_dir

        os.system("rm -rf " + logs_dir + '/train')
        os.system("rm -rf " + logs_dir + '/valid')
        os.system("mkdir " + logs_dir + '/train')
        os.system("mkdir " + logs_dir + '/valid')

        Vl.main(save_dir, prj_dir, 'LSTM', 'train', dev='/gpu:'+gpu_no)

        gs.freeze_graph(prj_dir + '/logs/LSTM', prj_dir + '/saved_model/graph/LSTM', 'model_1/soft_pred,model_1/raw_labels')

    print("done")
