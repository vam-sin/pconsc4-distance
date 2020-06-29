# libraries
import numpy as np 
import pandas as pd
import h5py
import pickle

train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

count = 0
for key in f_test['gdca']:
	mat = f_test['gdca'][key][()]
	a = mat.shape[0]

	count += a*a

print(count)
# Samples in Train: 
# 112239102
# 11223910
# Samples in Test: 8778266