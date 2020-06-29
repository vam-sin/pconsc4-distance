import numpy as np 
import h5py
from preprocess import get_datapoint
from model import unet
import pickle
import random
import keras
import keras.backend as K
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.utils import np_utils, plot_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization, Dense
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Conv2D, Conv1D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from one_ds import one_ds_generator
from model import unet

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


train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

num_classes = 7
train_gen = one_ds_generator(f_train, num_classes = num_classes)
test_gen = one_ds_generator(f_test, num_classes = num_classes)

inp = Input(shape = (140,))
x = Dense(100, activation='relu')(inp)
x = Dense(100, activation='relu')(x)
out = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=inp, outputs=out)
# model = unet(num_classes = num_classes)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
mcp_save = keras.callbacks.callbacks.ModelCheckpoint('feat_140.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]
model.fit_generator(train_gen, steps_per_epoch = 1120000, epochs = 100, validation_data = test_gen, validation_steps =  1120000, callbacks=callbacks_list)

# model = load_model('dense.h5')
# output = model.evaluate_generator(test_gen, 100000, verbose=1)
# print(output)

# Train: 0.94861
# Validation: 0.7875699996948242


