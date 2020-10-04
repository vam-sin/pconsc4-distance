# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
import h5py
from preprocess_pcons import get_datapoint
from model_resnet import resnet
import pickle
import random
import keras
from keras.models import load_model 
from sklearn.metrics import classification_report, confusion_matrix
import keras.backend as K
from keras import metrics
import tensorflow as tf
tf.random.set_seed(1234)

# GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf 

tf.keras.backend.clear_session()

config = ConfigProto()
config.gpu_options.allow_growth = True
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

LIMIT = 8 * 1024
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


def weighted_cce_3d(weights): 
  # shape of both (1, None, None, num_classes)
  def cce_3d(y_true, y_pred):
    y_pred = K.squeeze(y_pred, axis=0)
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.reshape(y_true, (-1, num_classes))

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True) # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    loss = K.sum(loss, axis=0)
      
    return loss

  return cce_3d

def jaccard_distance_loss(smooth=100):

  def jdl(y_true, y_pred):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    loss = (1 - jac) * smooth

    return loss

  return jdl 

def dice_coef_loss(smooth=1):
  
  def dice_coef(y_true, y_pred):
      intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
      loss = (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
      loss = 1 - loss 
      
      return loss

  return dice_coef

# functions
train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

num_classes = 7

if num_classes == 7:
  weights = [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
  weights = np.asarray(weights)

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
model = resnet(num_classes = num_classes+1)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('resnet_gpu.h5', save_best_only=True, monitor='val_loss', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

sgd = keras.optimizers.SGD(learning_rate = 1e-3)
adam = keras.optimizers.Adam(learning_rate = 1e-2)
rms = keras.optimizers.RMSprop(learning_rate = 1e-7)
# loss = weighted_cce_3d(weights)
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = [metrics.sparse_categorical_accuracy])
with tf.device('/gpu:0'):
  model.fit_generator(train_gen, epochs = 50, steps_per_epoch = 290, verbose=1, validation_data = test_gen, validation_steps = 210, callbacks = callbacks_list)


'''PPV:
FC DenseNet 103 Progress:
Only first Conv: 0.6724122766679188

Modified UNet Progress:
Added Extra Conv Layer in Add2DConv: 0.6227462202291915
'''

'''
Custom Loss Function:
Number of mismatches b/n y_pred and y_true
'''


