'''Tasks
1. Make train and test datasets - Test Done, Figure out a way to get train also.
2. Get the unet model - Done
3. Get training - 
Training on the test data just to check:

'''

# libraries
import numpy as np 
import h5py
from preprocess import get_datapoint
from model import unet
import pickle
import random

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
def generator(X, y, batch_size = 1):

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

# file
# train
# train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
# f_train = h5py.File(train_file_name, 'r')

# X_train = []
# y_train = []

# i = 0
# # print(len(f_train['gdca']))
# for key in f_train['gdca']:
# 	X_p, y_p = get_datapoint(f_train, key)
# 	X_train.append(X_p)
# 	y_train.append(y_p)
# 	print(i+1, ", Train: 2891")
# 	i += 1

# # pickle out
# filename = 'X_train.pickle'
# outfile = open(filename,'wb')
# pickle.dump(X_train, outfile)
# outfile.close()

# filename = 'y_train.pickle'
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
infile = open('X_test.pickle','rb')
X_test = np.asarray(pickle.load(infile))
infile.close()

infile = open('y_test.pickle','rb')
y_test = np.asarray(pickle.load(infile))
infile.close()

print("Loaded Test")

test_gen = generator(X_test, y_test)

# model
model = unet()
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
with tf.device('/cpu:0'):
	model.fit_generator(test_gen, epochs = 10, steps_per_epoch = 210, verbose=1)






