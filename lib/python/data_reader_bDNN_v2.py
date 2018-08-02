import numpy as np
import os
import glob
import utils
import scipy.io as sio

SAMPLE_RATE = 16000
num_features = 256

def frame_l2feature_l(data):
    iidx = 0
    Frame_iidx = 0
    wsize = int(SAMPLE_RATE * 25 * 0.001)
    wstep = int(SAMPLE_RATE * 10 * 0.001)

    nx = np.shape(data)[0]
    noverlap = wsize - wstep
    framelen = (nx - noverlap) // (wsize - noverlap)

    Detect = np.zeros(framelen)

    while True:

        if iidx + wsize < np.shape(data)[0]:
            TrueLabel_frame = data[iidx:iidx + wsize] * 10

        else:
            TrueLabel_frame = data[iidx:] * 10

        if sum(TrueLabel_frame) >= wsize / 2:
            TrueLabel_frame = 1
        else:
            TrueLabel_frame = 0

        if Frame_iidx >= np.shape(Detect)[0]:
            break

        Detect[Frame_iidx] = TrueLabel_frame
        iidx = iidx + wstep
        Frame_iidx = Frame_iidx + 1
        if iidx > np.shape(data)[0]:
            break
    return Detect

class DataReader(object):

    def __init__(self, input_dir, output_dir, norm_dir, w=19, u=9, name=None):
        # print(name + " data reader initialization...")
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._norm_dir = norm_dir
        self._input_file_list = sorted(glob.glob(input_dir + '/*.npy'))
        # self._input_spec_list = sorted(glob.glob(input_dir+'/*.txt'))
        self._output_file_list = sorted(glob.glob(output_dir + '/*.npy'))
        self._file_len = len(self._input_file_list)
        self._name = name
        assert self._file_len == len(self._output_file_list), "# input files and output file is not matched"
        self._w = w
        self._u = u
        self.eof = False
        self.file_change = False
        self.num_samples = 0

        self._inputs = 0
        self._outputs = 0

        self._epoch = 1
        self._num_file = 0
        self._start_idx = self._w

        # self._inputs = self._padding(
        #     self._read_input(self._input_file_list[self._num_file],
        #                      self._input_spec_list[self._num_file]), self._pad, self._w)
        #
        # self._outputs = self._padding(self._read_output(self._output_file_list[self._num_file]), self._pad, self._w)
        #
        # self.eof = False
        # self.file_change = False
        # self._outputs = self._outputs[0:self._inputs.shape[0]]
        # assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
        #     ("# samples is not matched between input: %d and output: %d files"
        #      % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))
        #
        # self.num_samples = np.shape(self._outputs)[0]

        norm_param = sio.loadmat(self._norm_dir+'/global_normalize_factor.mat')
        self.train_mean = norm_param['global_mean']
        self.train_std = norm_param['global_std']

        self.raw_inputs = 0  # adding part
        # print("Done.")
        # print("BOF : " + self._name + " file_" + str(self._num_file).zfill(2))

    def _binary_read_with_shape(self):
        pass

    @staticmethod
    def _read_input(input_file_dir):

        data = np.load(input_file_dir)  # (# total frame, feature_size)
        data = data[:, :num_features]
        # with open(input_spec_dir, 'r') as f:
        #     spec = f.readline()
        #     size = spec.split(',')
        # data = data.reshape((int(size[0]), int(size[1])), order='F')

        return data

    def get_cur_file_name(self):
        return self._input_file_list[self._num_file-1]

    @staticmethod
    def _read_output(output_file_dir):

        data = np.load(output_file_dir)
        # data = frame_l2feature_l(data)# data shape : (# total frame,)
        data = data.reshape(-1, 1)  # data shape : (# total frame, 1)

        return data


    @staticmethod
    def _padding(inputs, batch_size, w_val):
        pad_size = batch_size - inputs.shape[0] % batch_size

        inputs = np.concatenate((inputs, np.zeros((pad_size, inputs.shape[1]), dtype=np.float32)))

        window_pad = np.zeros((w_val, inputs.shape[1]))
        inputs = np.concatenate((window_pad, inputs, window_pad), axis=0)
        return inputs

    def next_batch(self, batch_size):

        self._inputs = self._read_input(self._input_file_list[self._num_file])
        self._outputs = self._read_output(self._output_file_list[self._num_file])
        assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
            ("# samples is not matched between input: %d and output: %d files"
             % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))
        self.num_samples = np.shape(self._outputs)[0]

        self._start_idx = self._w
        self.file_change = True
        self._num_file += 1

        if self._num_file > self._file_len - 1:
            self.eof = True
            self._num_file = 0

        # self._inputs = self._read_input(self._input_file_list[self._num_file],self._input_spec_list[self._num_file])
        # self._outputs = self._read_output(self._output_file_list[self._num_file])

        data_len = np.shape(self._inputs)[0]
        self._outputs = self._outputs[0:data_len, :]

        assert np.shape(self._inputs)[0] == np.shape(self._outputs)[0], \
            ("# samples is not matched between input: %d and output: %d files"
             % (np.shape(self._inputs)[0], np.shape(self._outputs)[0]))

        self.num_samples = np.shape(self._outputs)[0]

        inputs = self._inputs
        self.raw_inputs = inputs  # adding part
        inputs = self.normalize(inputs)
        inputs = utils.bdnn_transform(inputs, self._w, self._u)
        # inputs = inputs[self._w: -self._w, :]

        outputs = self._outputs
        outputs = utils.bdnn_transform(outputs, self._w, self._u)
        # outputs = outputs[self._w: -self._w, :]

        # self._start_idx += batch_size

        return inputs, outputs

    def normalize(self, x):
        x = (x - self.train_mean[:, :num_features])/self.train_std[:, :num_features]
        return x

    def reader_initialize(self):
        self._num_file = 0
        self._start_idx = 0
        self.eof = False

    def eof_checker(self):
        return self.eof

    def file_change_checker(self):
        return self.file_change

    def file_change_initialize(self):
        self.file_change = False

    def set_random_batch(self, batch_size):
        self._start_idx = np.maximum(0, np.random.random_integers(self.num_samples - batch_size))

    def get_file_len(self):
        return self._file_len


def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors."""
    # copied from TensorFlow tutorial
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
