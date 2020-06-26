# libraries
import numpy as np 
import h5py

# param
max_h = 496
max_w = 496

# functions
def pad(feature_map, w = 496, h = 496):
    result = np.zeros((w, h))
    result[:feature_map.shape[0],:feature_map.shape[1]] = feature_map
    
    return result

def get_datapoint(f, key):
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
    # pad
    num_cols_needed = orig_shape - len(seq[0])
    for i in range(num_cols_needed):
        placeholder = np.zeros((orig_shape,1))
        # print(placeholder.shape)
        seq = np.append(seq, placeholder, axis=1)

    seq = pad(seq)
    # print(seq.shape)

    group = f['part_entr']
    part_entr = group[key][()]
    # pad
    num_cols_needed = orig_shape - len(part_entr[0])
    for i in range(num_cols_needed):
        placeholder = np.zeros((orig_shape,1))
        # print(placeholder.shape)
        part_entr = np.append(part_entr, placeholder, axis=1)

    part_entr = pad(part_entr)
    # print(part_entr.shape)

    group = f['self_info']
    self_info = group[key][()]
    # pad
    num_cols_needed = orig_shape - len(self_info[0])
    for i in range(num_cols_needed):
        placeholder = np.zeros((orig_shape,1))
        # print(placeholder.shape)
        self_info = np.append(self_info, placeholder, axis=1)

    self_info = pad(self_info)
    # print(self_info.shape)

    X = np.stack((gdca, cross_h, nmi_corr, mi_corr, seq, part_entr, self_info), axis=2)
    # print(X.shape)

    # y
    group = f['dist']
    max_d = 100.0
    num_classes = 11

    y = []
    dist = group[key][()]
    for i in dist:
        y_r = []
        for j in i:
            dis_class = np.zeros(num_classes)
            if str(j) == 'inf' or j > max_d:
                dis_class[num_classes-1] = 1.0
            else:
                dis_class[int(j/(num_classes-1))] = 1.0
            y_r.append(dis_class)

        y_r = np.asarray(y_r)
        y_r = list(y_r)
        y.append(y_r)

    y = np.asarray(y)
    padded_y = np.stack((pad(y[:,:,0]), pad(y[:,:,1]), pad(y[:,:,2]), pad(y[:,:,3]), pad(y[:,:,4]), pad(y[:,:,5]), pad(y[:,:,6]), pad(y[:,:,7]), pad(y[:,:,8]), pad(y[:,:,9]), pad(y[:,:,10])), axis=2)

    padded_y = np.asarray(padded_y)
    # print(padded_y.shape)
    print("Preprocessed data point")
    return X, padded_y


if __name__ == '__main__':

    file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
    f = h5py.File(file_name, 'r')
    
    datx, daty = get_datapoint(f, '4J32B.hhE0')

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