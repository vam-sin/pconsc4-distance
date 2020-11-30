import h5py
import numpy as np
from keras.utils import to_categorical 
import keras.backend as K

def pad(x, pad_even, depth=4):
    divisor = np.power(2, depth)
    remainder = x.shape[0] % divisor
    # no padding
    if not pad_even:
        return x
    # no padding because already of even shape
    elif pad_even and remainder == 0:
        return x
    # add zero rows after 1D feature
    elif len(x.shape) == 2:
        return np.pad(x, [(0, divisor - remainder), (0, 0)], "constant")
    # add zero columns and rows after 2D feature
    elif len(x.shape) == 3:
        return np.pad(x, [(0, divisor - remainder), (0, divisor - remainder), (0, 0)], "constant")

def get_datapoint(h5file, sequence_predictions, key, num_classes, pad_even=True):
    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr']
    label = "dist"
    # key_lst = list(f['gdca'].keys())
    x_i_dict = {}
    for feat in feat_lst:
        if feat in ['sep']:
            x_i = np.array(range(L))
            x_i = np.abs(np.add.outer(x_i, -x_i))
        elif feat in ['gneff']:
            x_i = h5file[feat][key][()]
            x_i = log10(x_i)
            # x_i = np.outer(x_i, x_i)
        elif feat in ['plm_J']:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        elif feat in ['cross_h', 'mi_corr', 'nmi_corr', 'plm', 'gdca']:
            x_i = h5file[feat][key][()]
            if feat == 'gdca':
                x_i = (x_i + 1.298669)/(5.624298 + 1.298669)
            elif feat == 'mi_corr':
                x_i = (x_i + 2.266821)/(2.7405028 + 2.266821) 
            elif feat == 'nmi_corr':
                x_i = (x_i + 1.0450006)/(0.9999157 + 1.0450006) 
            elif feat == 'cross_h':
                x_i = (x_i - 2e-07)/(5.710513 - 2e-07)
            L = x_i.shape[0]
            x_i = x_i[..., None]
            # print(x_i.shape)
        else:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        x_i = pad(x_i, pad_even)
        # print(x_i.shape)
        x_i_dict[feat] = x_i[None, ...]

    seq = h5file["seq"][key][()]
    seq = pad(seq, pad_even)
    x_i_dict["seq"] = seq[None, ...]

    part_entr = h5file["part_entr"][key][()]
    part_entr = pad(part_entr, pad_even)
    x_i_dict["part_entr"] = part_entr[None, ...]

    self_info = h5file["self_info"][key][()]
    self_info = pad(self_info, pad_even)
    x_i_dict["self_info"] = self_info[None, ...]

    
    x_i_dict["seq_input"] = sequence_predictions[key]
    # print(x_i_dict["seq_input"].shape)

    y = h5file[label][key][()]
    y = y[..., None]  # reshape from (L,L) to (L,L,1)
    y = pad(y, pad_even)
    y = y[None, ...]
    # y = y[None, ...]

    if num_classes == 7:
        bins = [4, 6, 8, 10, 12, 14]
        no_bins = 7
        y[y == np.inf] = 15.0
        y[y >= 15.0] = 15.0
        weights = [0.3942871, 0.3161621, 0.3942871, 0.4846191, 0.802002,  0.7897949, 0.68188477]
    elif num_classes == 12:
        bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        no_bins = 12
    elif num_classes == 26:
        bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
        no_bins = 26
    elif num_classes == 37:
        bins = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5]
        no_bins = 37
        # print(len(bins))
        # y = np.squeeze(y, axis = 2)
        # # print(y.shape)
        # # last class is no contact
        # # for i in range(y.shape[0]):
        # #     for j in range(y.shape[0]):
        # #         if y[i][j] < 2 or y[i][j] >= bins[len(bins)-1]:
        # #             y[i][j] = 20.25
        # y = np.searchsorted(bins, y, side='right')
        #         # else:
        #         #     for k in range(len(bins)-1):
        #         #         if y[i][j] >= bins[k] and y[i][j] < bins[k+1]:
        #         #             y[i][j] = k

        # # y[y < bins[0] or y >= bins[len(bins)-1]] = 36

        # y = to_categorical(y, num_classes = no_bins+1)

    y = np.searchsorted(bins, y)
    # y = y.astype('int')
    # weight_matrix = np.zeros((y.shape))
    # for i in range(y.shape[0]):
    #     for j in range(y.shape[1]):
    #         for k in range(y.shape[2]):
    #             weight_matrix[i][j][k][0] = weights[y[i][j][k][0]]
    # y = np.squeeze(y, axis=3)
    y = to_categorical(y, num_classes = no_bins)

    return x_i_dict, y

def get_datapoint_align(h5file, feature_dict, key, num_classes, pad_even=True):
    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr']
    label = "dist"
    # key_lst = list(f['gdca'].keys())
    x_i_dict = {}
    for feat in feat_lst:
        if feat in ['sep']:
            x_i = np.array(range(L))
            x_i = np.abs(np.add.outer(x_i, -x_i))
        elif feat in ['gneff']:
            x_i = h5file[feat][key][()]
            x_i = log10(x_i)
            # x_i = np.outer(x_i, x_i)
        elif feat in ['plm_J']:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        elif feat in ['cross_h', 'mi_corr', 'nmi_corr', 'plm', 'gdca']:
            x_i = feature_dict[feat]
            if feat == 'gdca':
                x_i = (x_i + 1.298669)/(5.624298 + 1.298669)
            elif feat == 'mi_corr':
                x_i = (x_i + 2.266821)/(2.7405028 + 2.266821) 
            elif feat == 'nmi_corr':
                x_i = (x_i + 1.0450006)/(0.9999157 + 1.0450006) 
            elif feat == 'cross_h':
                x_i = (x_i - 2e-07)/(5.710513 - 2e-07)
            L = x_i.shape[0]
            x_i = x_i[..., None]
            x_i = np.squeeze(x_i, axis = 3)
            x_i = np.squeeze(x_i, axis = 0)
            # print(x_i.shape)
        else:
            x_i = feature_dict[feat]
            L = x_i.shape[0]
        # x_i = pad(x_i, pad_even)
        # print(x_i.shape)
        x_i_dict[feat] = x_i[None, ...]

    seq = feature_dict["seq"]
    # seq = pad(seq, pad_even)
    x_i_dict["seq"] = seq

    part_entr = feature_dict["part_entr"]
    # part_entr = pad(part_entr, pad_even)
    x_i_dict["part_entr"] = part_entr

    self_info = feature_dict["self_info"]
    # self_info = pad(self_info, pad_even)
    x_i_dict["self_info"] = self_info

    y = h5file[label][key][()]
    y = y[..., None]  # reshape from (L,L) to (L,L,1)
    y = pad(y, pad_even)
    # y = y[None, ...]

    if num_classes == 7:
        bins = [4, 6, 8, 10, 12, 14]
        # [0, 4-6, 6-8, 8-10, 10-12, 12-14, 14+]
        no_bins = 7
    elif num_classes == 12:
        bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        no_bins = 12
    elif num_classes == 26:
        bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
        no_bins = 26
    elif num_classes == 37:
        bins = [2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20]
        no_bins = 37

    # y = np.searchsorted(bins, y)
    # print(y.shape)
    # y = to_categorical(y, num_classes = no_bins)
    # print(y.shape)

    return x_i_dict, y

def mean_squared_error(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.squeeze(y_true, axis=2)
    y_pred = K.squeeze(y_pred, axis=0)
    y_pred = K.squeeze(y_pred, axis=2)
    return K.mean(K.square(y_pred - y_true))

if __name__ == '__main__':

    file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
    f = h5py.File(file_name, 'r')
    sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()
    
    # num_classes = 7, 12, 26
    key_lst = list(f["gdca"].keys())
    X, y = get_datapoint(f, sequence_predictions, key_lst[0], 7)
    print(y.shape)
    # y_t = y.copy()
    # y_t = y_t + 2
    # print(mean_squared_error(y_t, y))
    
    # print(len(key_lst))
    # feat = "cross_h"
    # summation = 0.0 
    # max_ = -10000
    # min_ = 10000
    # for i in key_lst:
    #     print(i)
    #     X = f[feat][i][()]
    #     max_ = max(np.max(X), max_)
    #     min_ = min(np.min(X), min_)
    #     summation += np.sum(X)
    # print(summation/len(key_lst), max_, min_)




    # for i in range(len(y_values)):
    #     for j in range(len(y_values)):
    #         print(y[0][i][j], y_values[i][j])
'''
Think of Normalizing data
gdca: 5.624298 -1.298669
mi_corr: 2.7405028 -2.266821
nmi_corr: 0.9999157 -1.0450006
cross_h: 5.710513 2e-07
// DId
'''