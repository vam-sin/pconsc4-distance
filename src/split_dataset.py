# proper 80-20 split. 

import numpy as np 
import h5py
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# GPU
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf 

tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 4 * 1024
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

from keras import backend as K

# functions
train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

total_keys = []

train_keys = list(f_train["gdca"].keys())
test_keys = list(f_test["gdca"].keys())
print(len(train_keys), len(test_keys))

for i in train_keys:
	total_keys.append(i)

for i in test_keys:
	total_keys.append(i)

print(len(total_keys))
total_keys = shuffle(total_keys, random_state = 42)
keys_train, keys_test = train_test_split(total_keys, test_size=0.2, random_state=42)
print(len(keys_train), len(keys_test))
