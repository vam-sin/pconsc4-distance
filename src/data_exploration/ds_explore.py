# libraries
import numpy as np 

def pad(x, pad_even, depth = 4):

    divisor = np.power(2, depth)
    remainder = x.shape[0] % divisor

    if not pad_even:
        return x
    elif pad_even and remainder == 0:
        return x
    elif len(x.shape) == 2:
        return np.pad(x, [(0, divisor - remainder), (0,0)], "constant")
    elif len(x.shape) == 3:
        return np.pad(x, [(0, divisor - remainder), (0, divisor - remainder), (0,0)], "constant")

def get_datapoint(h5file, feat_lst, label, binary_cutoffs, key, pad_even = False):

    x_i_dict = {}
    
    for feat in feat_lst:
        if feat in ['sep']:
            x_i = np.array(range(L))
            x_i = np.abs(np.add.outer(x_i, -x_i))
        elif feat in ['gneff']:
            x_i = h5file[feat][key][()]
            x_i = log10(x_i)
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

    mask = h5file[label][key][()]
    mask = mask != 0.0
    mask = mask[..., None]
    mask = pad(mask, pad_even)
    mask = mask[None, ...]

    y = h5file[label][key][()]
    y = y[..., None]
    y = pad(y, pad_even)
    y = y[None, ...]

    y_binary_dict = {}
    y_dist = h5file["dist"][key][()]

    for d in binary_cutoffs:
        y_binary = y_dist < d
        y_binary = y_binary[..., None]
        y_binary = pad(y_binary, pad_even)
        y_binary = y_binary[None, ...]
        y_binary_dict[d] = y_binary

    return x_i_dict, mask, y, y_binary_dict, L

file_name = "../../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"

import h5py

f = h5py.File(file_name, 'r')

# print(f)

# for key in f.keys():
#     print(key) #Names of the groups in HDF5 file.

# #Get the HDF5 group
group = f['part_entr']

#Checkout what keys are inside that group.
# for key in group.keys():
#     data = group[key].value
#     print(data.shape, key)
# 	#Do whatever you want with 

d = group['4J32B.hhE0'][()]
print(d.shape)

# #After you are done
# f.close()
# feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr', 'seq', 'part_entr', 'self_info']
# label = "dist"
# binary_cutoffs = [6, 8, 10]
# key = "4J32B.hhE0"
# pad_even = True

# x_i_dict, mask, y, y_binary_dict, L = get_datapoint(f, feat_lst, label, binary_cutoffs, key, pad_even = True)

# '''
# x_i_dist has 7 arrays belonging to the 7 features.
# mask and y are the same but mask has false whereas y has 0
# y_binary_dict has three arrays which tell the contact prediction in each of the three thresholds.
# '''
# print(y[0][0], y.shape)