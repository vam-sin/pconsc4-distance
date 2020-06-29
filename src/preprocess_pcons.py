import h5py
import numpy as np

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

def get_datapoint(h5file, feat_lst, label, binary_cutoffs, key, pad_even=True):
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

    mask = h5file[label][key][()]
    mask = mask != 0.0
    mask = mask[..., None]  # reshape from (L,L) to (L,L,1)
    mask = pad(mask, pad_even)
    mask = mask[None, ...]  # reshape from (L,L,1) to (1,L,L,1)

    y = h5file[label][key][()]
    y = y[..., None]  # reshape from (L,L) to (L,L,1)
    y = pad(y, pad_even)
    y = y[None, ...]

    y_binary_dict = {}
    y_dist = h5file["dist"][key][()]
    for d in binary_cutoffs:
        y_binary = y_dist < d
        y_binary = y_binary[..., None]  # reshape from (L,L) to (1,L,L,1)
        y_binary = pad(y_binary, pad_even)
        y_binary = y_binary[None, ...]
        y_binary_dict[d] = y_binary

    return x_i_dict, mask, y, y_binary_dict, L