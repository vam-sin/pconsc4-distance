import h5py
import numpy as np
from keras.utils import to_categorical 

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

def get_datapoint(h5file, key, pad_even=True):
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
            L = x_i.shape[0]
            x_i = x_i[..., None]
        else:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        x_i = pad(x_i, pad_even)
        x_i_dict[feat] = x_i[None, ...]

    y = h5file[label][key][()]
    y = y[..., None]  # reshape from (L,L) to (L,L,1)
    y = pad(y, pad_even)
    y = y[None, ...]

    bins = [4, 6, 8, 10, 12, 14]
    no_bins = 7
    y = np.searchsorted(bins, y)
    y = to_categorical(y, num_classes = no_bins)

    # batch_features_dict = {}
    # batch_features_dict["input_1"] = x_i_dict["gdca"]
    # batch_features_dict["input_2"] = x_i_dict["cross_h"]
    # batch_features_dict["input_3"] = x_i_dict["mi_corr"]
    # batch_features_dict["input_4"] = x_i_dict["nmi_corr"]

    return x_i_dict, y

if __name__ == '__main__':

    file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
    f = h5py.File(file_name, 'r')
    
    # num_classes = 7, 12, 26
    feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr']
    label = "dist"
    key_lst = list(f['gdca'].keys())
    x_i_dict, y = get_datapoint(f, feat_lst, label, key_lst[2])
    bins = [4, 6, 8, 10, 12, 14]
    no_bins = 7
    batch_labels_dict = {}
    print(y)
    y = np.searchsorted(bins, y)
    print(y)
    y = to_categorical(y, num_classes = no_bins)
    batch_labels_dict["out_%s_mask" % label] = y

    print(y.shape)