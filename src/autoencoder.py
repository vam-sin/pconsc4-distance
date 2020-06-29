# libraries
import keras
import keras.backend as K
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.utils import np_utils, plot_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization, Flatten
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Conv2D, Conv1D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose

import numpy as np 
import h5py
from preprocess import get_datapoint_reg
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

# model definition function
# param
dropout = 0.1
smooth = 1.
act = ELU
init = "he_normal"
reg_strength = float(10**-12)
reg = l2(reg_strength)
num_filters = 1

def add_2D_conv(x, filters, kernel_size, data_format="channels_last", padding="same", depthwise_initializer=init, pointwise_initializer=init, depthwise_regularizer=reg, 
        pointwise_regularizer=reg):
	x = Conv2D(num_filters, kernel_size, data_format=data_format, padding=padding)(x)
	# x = Dropout(dropout)(x)
	x = act()(x)
	# x = BatchNormalization()(x)

	return x

def unet(num_classes):
	inp = Input(shape = (496, 496, 5))

	#Downsampling
	unet = Conv2D(64, 3, data_format="channels_last", padding="same")(inp)
	mid = MaxPooling2D(pool_size=(16, 16), data_format = "channels_last", padding='same')(unet)

	unet = UpSampling2D((16,16), data_format = "channels_last")(mid)
	unet = Conv2D(64, 3, data_format="channels_last", padding="same")(unet)

	output = Conv2D(5, 7, activation ="linear", data_format = "channels_last", 
	        padding = "same")(unet)

	model = Model(inputs = inp, outputs = output)
	auto = Model(inputs = inp, outputs = mid)
	print(model.summary())

	return model, auto

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

      X, y = get_datapoint_reg(h5file, key, num_classes)

      batch_features_dict = np.expand_dims(X, axis = 0)

      batch_labels_dict = np.expand_dims(y, axis = 0)

      i += 1

      yield batch_features_dict, batch_features_dict

if __name__ == '__main__':
	model, auto = unet(num_classes = 7)
	train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
	f_train = h5py.File(train_file_name, 'r')

	test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
	f_test = h5py.File(test_file_name, 'r')

	num_classes = 7
	train_gen = generator_from_file(f_train, num_classes = num_classes)
	test_gen = generator_from_file(f_test, num_classes = num_classes)


	mcp_save = keras.callbacks.callbacks.ModelCheckpoint('auto.h5', save_best_only=True, monitor='mse', verbose=1)
	reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='mse', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	callbacks_list = [reduce_lr, mcp_save]

	model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])
	# with tf.device('/cpu:0'):
	model.fit_generator(train_gen, epochs = 10, steps_per_epoch = 320, verbose=1, callbacks = callbacks_list)

	new_X = auto.predict_generator(train_gen)





