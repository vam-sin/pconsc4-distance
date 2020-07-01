# libraries
import numpy as np 
import h5py
from preprocess_pcons import get_datapoint
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

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

num_classes = 26

def generator_from_file(h5file, num_classes, batch_size=1):
  key_lst = list(h5file['gdca'].keys())
  random.shuffle(key_lst)
  i = 0

  while True:
      # TODO: different batch sizes with padding to max(L)
      # for i in range(batch_size):
      # index = random.randint(1, len(features)-1)
      if i == len(key_lst):
          random.shuffle(key_lst)
          i = 0

      key = key_lst[i]
      # print(key)

      X, y = get_datapoint(h5file, key, num_classes)

      batch_labels_dict = {}
      batch_labels_dict["out_dist"] = y

      i += 1

      yield X, batch_labels_dict

train_gen = generator_from_file(f_train, num_classes = num_classes)
test_gen = generator_from_file(f_test, num_classes = num_classes)

# model
model = unet(num_classes = num_classes)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('unet.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
with tf.device('/cpu:0'):
  model.fit_generator(train_gen, epochs = 10, steps_per_epoch = 290, verbose=1, validation_data = test_gen, validation_steps = 210, callbacks = callbacks_list)






