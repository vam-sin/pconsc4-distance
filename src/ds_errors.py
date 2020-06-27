import h5py

train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

group_train = f_train['gdca']
group_test = f_test['gdca']

keys_train = []
for key in group_train:
	keys_train.append(key)

keys_test = []
for key in group_test:
	keys_test.append(key)

for i in keys_test:
	if i in keys_train:
		print(i)

# no common elements in test and train

