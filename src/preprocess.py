# libraries
import numpy as np 
import h5py

# param
max_h = 496
max_w = 496

# functions
def pad(x, pad_even=True, depth=4):
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

def get_datapoint(f, key, num_classes):
    # X
    group = f['gdca']
    gdca = group[key][()]
    orig_shape = len(gdca)
    gdca = pad(gdca)
    print(gdca.shape)

    group = f['cross_h']
    cross_h = group[key][()]
    cross_h = pad(cross_h)
    # print(cross_h.shape)

    group = f['nmi_corr']
    nmi_corr = group[key][()]
    nmi_corr = pad(nmi_corr)
    # print(nmi_corr.shape)

    group = f['mi_corr']
    mi_corr = group[key][()]
    mi_corr = pad(mi_corr)
    # print(mi_corr.shape)

    group = f['seq']
    seq = group[key][()]
    group = f['part_entr']
    part_entr = group[key][()]
    group = f['self_info']
    self_info = group[key][()]

    feat_68 = []
    for i in range(len(part_entr)):
        row = []
        for j in seq[i]:
            row.append(j)
        for j in part_entr[i]:
            row.append(j)
        for j in self_info[i]:
            row.append(j)
        
        feat_68.append(row)

    feat_68 = np.asarray(feat_68)

    num_cols_needed = orig_shape - len(part_entr[0])
    for i in range(num_cols_needed):
        placeholder = np.zeros((orig_shape,1))
        # print(placeholder.shape)
        part_entr = np.append(feat_68, placeholder, axis=1)

    feat_68 = pad(feat_68)
    # print(self_info.shape)

    X = np.stack((gdca, cross_h, nmi_corr, mi_corr, feat_68), axis=2)
    del gdca, cross_h, nmi_corr, mi_corr, feat_68
    # print(X.shape)

    # y
    group = f['dist']
    y = []
    dist = group[key][()]

    if num_classes == 7:
        bins = [2, 5, 7, 9, 11, 13, 15]
    elif num_classes == 12:
        bins = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
    elif num_classes == 26:
        bins = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]

    max_d = bins[len(bins)-1]

    for i in dist:
        y_r = []
        for j in i:
            dis_class = np.zeros(num_classes)
            if str(j) == 'inf' or j > max_d:
                dis_class[num_classes-1] = 1.0

            for k in range(len(bins)-1):
                if j >= bins[k] and j < bins[k+1]:
                    dis_class[k] = 1.0

            y_r.append(dis_class)

        y_r = np.asarray(y_r)
        y_r = list(y_r)
        y.append(y_r)

    y = np.asarray(y)

    if num_classes == 7:
        padded_y = np.stack(( pad(y[:,:,0]), pad(y[:,:,1]), pad(y[:,:,2]), pad(y[:,:,3]), pad(y[:,:,4]), pad(y[:,:,5]), pad(y[:,:,6]) ), axis=2)
    elif num_classes == 12:
        padded_y = np.stack(( pad(y[:,:,0]), pad(y[:,:,1]), pad(y[:,:,2]), pad(y[:,:,3]), pad(y[:,:,4]), pad(y[:,:,5]), pad(y[:,:,6]), pad(y[:,:,7]), pad(y[:,:,8]), pad(y[:,:,9]), pad(y[:,:,10]), pad(y[:,:,11]) ), axis=2)
    elif num_classes == 26:
        padded_y = np.stack(( pad(y[:,:,0]), pad(y[:,:,1]), pad(y[:,:,2]), pad(y[:,:,3]), pad(y[:,:,4]), pad(y[:,:,5]), pad(y[:,:,6]), pad(y[:,:,7]), pad(y[:,:,8]), pad(y[:,:,9]), pad(y[:,:,10]), pad(y[:,:,11]), pad(y[:,:,12]), pad(y[:,:,13]), pad(y[:,:,14]), pad(y[:,:,15]), pad(y[:,:,16]), pad(y[:,:,17]), pad(y[:,:,18]), pad(y[:,:,19]), pad(y[:,:,20]), pad(y[:,:,21]), pad(y[:,:,22]), pad(y[:,:,23]), pad(y[:,:,24]), pad(y[:,:,25]) ), axis=2)


    padded_y = np.asarray(padded_y)
    del y
    # print(padded_y.shape)
    # print("Preprocessed data point")
    return X, padded_y

def get_datapoint_reg(f, key, num_classes):
    # X
    group = f['gdca']
    gdca = group[key][()]
    orig_shape = len(gdca)
    gdca = pad(gdca)
    # print(gdca.shape)

    group = f['cross_h']
    cross_h = group[key][()]
    cross_h = pad(cross_h)
    # print(cross_h.shape)

    group = f['nmi_corr']
    nmi_corr = group[key][()]
    nmi_corr = pad(nmi_corr)
    # print(nmi_corr.shape)

    group = f['mi_corr']
    mi_corr = group[key][()]
    mi_corr = pad(mi_corr)
    # print(mi_corr.shape)

    group = f['seq']
    seq = group[key][()]
    group = f['part_entr']
    part_entr = group[key][()]
    group = f['self_info']
    self_info = group[key][()]

    feat_68 = []
    for i in range(len(part_entr)):
        row = []
        for j in seq[i]:
            row.append(j)
        for j in part_entr[i]:
            row.append(j)
        for j in self_info[i]:
            row.append(j)
        
        feat_68.append(row)

    feat_68 = np.asarray(feat_68)

    num_cols_needed = orig_shape - len(part_entr[0])
    for i in range(num_cols_needed):
        placeholder = np.zeros((orig_shape,1))
        # print(placeholder.shape)
        part_entr = np.append(feat_68, placeholder, axis=1)

    feat_68 = pad(feat_68)
    # print(self_info.shape)

    X = np.stack((gdca, cross_h, nmi_corr, mi_corr, feat_68), axis=2)
    del gdca, cross_h, nmi_corr, mi_corr, feat_68
    # print(X.shape)

    # y
    group = f['dist']
    y = []
    dist = group[key][()]
    y = pad(dist)
    y = np.expand_dims(y, axis=2)

    # print(padded_y.shape)
    # print("Preprocessed data point")
    return X, y


if __name__ == '__main__':

    file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
    f = h5py.File(file_name, 'r')
    
    # num_classes = 7, 12, 26
    datx, daty = get_datapoint(f, '4J32B.hhE0', 7)

    print(daty.shape, datx.shape)

'''
Keys:
cross_h
dist
gdca
gneff
mi
mi_corr
nmi
nmi_corr
part_entr
plm
rsa
self_info
seq
ss
sscore
sscore10
sscore6

# Input Format:
Each of the different features will form a channel. 
Features Used: gdca, cross_h, mi_corr, nmi_corr, part_entr, seq, self_info
Shape: [num_residues, num_residues, 7]

# Tasks:
Input Features:
1. Make a function to convert each datapoint into input shape: [num_residues, num_residues, 7] - Done
2. Make a function to pad to them to the same size. - Done
Max Height: 494 (Test), 350 (Train)

Output Features:
1. Convert each distance value in the "dist" matrix into a bin value - Done (Still have to make for variable number of classes)
0-100 - 10 classes
100 + 11th class
Shape: [494, 494, num_classes]
'''