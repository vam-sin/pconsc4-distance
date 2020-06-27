'''Tasks
1. Make train and test datasets - Test Done, Figure out a way to get train also.
2. Get the unet model - Done
3. Get training - 
Training on the test data just to check:
loss: 6.7063e-04 - accuracy: 0.2352
'''

# libraries
import numpy as np 
import h5py
from preprocess import get_datapoint
from model import unet
import pickle
import random
import keras

# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf 

tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 3 * 1024
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=LIMIT)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

# functions
train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

def test_generator(X, y, batch_size = 1):

    index_lst = list(range(len(X)))
    random.shuffle(index_lst)
    i = 0

    while True:

        if i == len(index_lst):
            random.shuffle(index_lst)
            i = 0

        index = index_lst[i]

        batch_features_dict = np.expand_dims(X[0], axis = 0)

        batch_labels_dict = np.expand_dims(y[0], axis = 0)

        i += 1

        yield batch_features_dict, batch_labels_dict

def train_generator(num_classes, batch_size = 1):

	key_lst = []

	for key in f_train['gdca']:
		key_lst.append(key)

	index_lst = list(range(len(key_lst)))
	random.shuffle(index_lst)

	i = 0

	while True:

		if i == len(index_lst):
			random.shuffle(index_lst)
			i = 0

		index = index_lst[i]

		X, y = get_datapoint(f_train, key_lst[index], num_classes)

		# print(X.shape, y.shape)

		batch_features_dict = np.expand_dims(X, axis = 0)

		batch_labels_dict = np.expand_dims(y, axis = 0)

		i += 1

		yield batch_features_dict, batch_labels_dict

# file
# train

# train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
# f_train = h5py.File(train_file_name, 'r')

# X_train = []
# y_train = []

# key_lst = []

# for key in f_train['gdca']:
# 	key_lst.append(key)

# # print(len(f_train['gdca']))
# for i in range(len(key_lst)):
# 	X_p, y_p = get_datapoint(f_train, key_lst[i])
# 	X_train.append(X_p)
# 	y_train.append(y_p)
# 	print(i+1, ", Train: 2891")
	# if i+1 == 300:
	# 	break

# # pickle out
# filename = 'X_train_1.pickle'
# outfile = open(filename,'wb')
# pickle.dump(X_train, outfile)
# outfile.close()

# filename = 'y_train_1.pickle'
# outfile = open(filename,'wb')
# pickle.dump(y_train, outfile)
# outfile.close()

# pickle in
# infile = open('X_train.pickle','rb')
# X_test = np.asarray(pickle.load(infile))
# infile.close()

# infile = open('y_train.pickle','rb')
# y_test = np.asarray(pickle.load(infile))
# infile.close()

# test
# test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
# f_test = h5py.File(test_file_name, 'r')

# X_test = []
# y_test = []

# i = 0
# # print(len(f_test['gdca']))
# for key in f_test['gdca']:
# 	X_p, y_p = get_datapoint(f_test, key)
# 	X_test.append(X_p)
# 	y_test.append(y_p)
# 	print(i+1, ", Test: 210")
# 	i += 1

# # pickle out
# filename = 'X_test.pickle'
# outfile = open(filename,'wb')
# pickle.dump(X_test, outfile)
# outfile.close()

# filename = 'y_test.pickle'
# outfile = open(filename,'wb')
# pickle.dump(y_test, outfile)
# outfile.close()

# pickle in
# infile = open('X_test.pickle','rb')
# X_test = np.asarray(pickle.load(infile))
# infile.close()

# infile = open('y_test.pickle','rb')
# y_test = np.asarray(pickle.load(infile))
# infile.close()

# print("Loaded Test")

# test_gen = test_generator(X_test, y_test)
num_classes = 7
train_gen = train_generator(num_classes = num_classes)

# # model
model = unet(num_classes = num_classes)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('unet.h5', save_best_only=True, monitor='accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
with tf.device('/cpu:0'):
	model.fit_generator(train_gen, epochs = 10, steps_per_epoch = 2891, verbose=1, callbacks = callbacks_list)






