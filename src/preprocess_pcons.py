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

def get_datapoint(h5file, key, num_classes, pad_even=True):
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
            print(x_i.shape)
        else:
            x_i = h5file[feat][key][()]
            L = x_i.shape[0]
        x_i = pad(x_i, pad_even)
        print(x_i.shape)
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

    y = h5file[label][key][()]
    y = y[..., None]  # reshape from (L,L) to (L,L,1)
    y = pad(y, pad_even)
    y = y[None, ...]

    if num_classes == 7:
        bins = [4, 6, 8, 10, 12, 14]
        no_bins = 7
    elif num_classes == 12:
        bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        no_bins = 12
    elif num_classes == 26:
        bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
        no_bins = 26

    y = np.searchsorted(bins, y)
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
    y = y[None, ...]

    if num_classes == 7:
        bins = [4, 6, 8, 10, 12, 14]
        no_bins = 7
    elif num_classes == 12:
        bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        no_bins = 12
    elif num_classes == 26:
        bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
        no_bins = 26

    y = np.searchsorted(bins, y)
    y = to_categorical(y, num_classes = no_bins)

    return x_i_dict, y

if __name__ == '__main__':

    file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
    f = h5py.File(file_name, 'r')
    
    # num_classes = 7, 12, 26
    key_lst = list(f["gdca"].keys())
    print(key_lst)
    X, y = get_datapoint(f, key_lst[0], 7)
    print(y, f["dist"][key_lst[0]][()].shape)