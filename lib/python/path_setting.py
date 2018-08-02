
class PathSetting(object):

    def __init__(self, prj_dir, model, save_dir):
        train_dir = save_dir + '/train'
        valid_dir = save_dir + '/test'
        logs_dir = prj_dir + '/logs/' + model

        self.logs_dir = logs_dir
        self.initial_logs_dir = logs_dir
        self.input_dir = train_dir+'/feature_mrcg'
        self.output_dir = train_dir+'/label'
        self.norm_dir = train_dir+'/feature_mrcg/global_normalization'
        self.valid_file_dir = valid_dir

