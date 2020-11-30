import h5py
import numpy as np
from split_preprocess_pcons import get_datapoint

file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f = h5py.File(file_name, 'r')
sequence_predictions = np.load('sequence_predictions.npy', allow_pickle='TRUE').item()

# num_classes = 7, 12, 26
ds = []
key_lst = list(f["gdca"].keys())
for k in range(10):
    X, y = get_datapoint(f, sequence_predictions, key_lst[k], 7)
    y = f["dist"][key_lst[k]][()]
    # y = np.squeeze(y, axis=0)
    print(k+1, len(key_lst))
    # bins = [4, 6, 8, 10, 12, 14]
    # no_bins = 7
    # count = 0
    for i in range(y.shape[0]-64):
        for j in range(y.shape[1]-64):
            x_64 = {}
            x_64["dist"] = y[np.ix_(range(i, i+64), range(j, j+64))]
            x_64["gdca"] = X["gdca"][np.ix_(range(i, i+64), range(j, j+64))]
            x_64["cross_h"] = X["cross_h"][np.ix_(range(i, i+64), range(j, j+64))]
            x_64["mi_corr"] = X["mi_corr"][np.ix_(range(i, i+64), range(j, j+64))]
            x_64["nmi_corr"] = X["nmi_corr"][np.ix_(range(i, i+64), range(j, j+64))]
            seq = np.squeeze(X["seq_input"], axis=0)
            x_64["seq_input"] = seq[np.ix_(range(i, i+64))]
            # y_prime = np.searchsorted(bins, y_prime)
            # print(i, j, x_64["dist"].shape, x_64["gdca"].shape, x_64["cross_h"].shape, x_64["mi_corr"].shape, x_64["nmi_corr"].shape, x_64["seq_input"].shape)
            ds.append(x_64)

np.save('ds_64.npy', ds) 