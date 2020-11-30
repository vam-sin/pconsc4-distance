# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
import h5py
from preprocess_pcons import get_datapoint
from model_actual_resnet import actual_resnet
import pickle
import random
import keras
from keras.models import load_model 
from sklearn.metrics import classification_report, confusion_matrix
import keras.backend as K
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_addons as tfa
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


def dice_coef_9cat(y_true, y_pred, smooth=1e-7):
    '''
    Dice coefficient for 10 categories. Ignores background pixel label 0
    Pass to model as metric during compile statement
    '''
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, 'int32'), num_classes=7)[...,1:])
    y_pred_f = K.flatten(y_pred[...,1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

def dice_coef_9cat_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - dice_coef_9cat(y_true, y_pred)

def focal_loss(gamma=2., alpha=.25):
  # focal loss
  def focal_loss_fixed(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

  return focal_loss_fixed

def mean_squared_error(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.squeeze(y_true, axis=2)
    y_pred = K.squeeze(y_pred, axis=0)
    y_pred = K.squeeze(y_pred, axis=2)
    return K.sum(K.mean(K.square(y_pred - y_true), axis=-1))

# functions
train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

# Proper 80-20 split
total_keys = []

train_keys = list(f_train["gdca"].keys())
test_keys = list(f_test["gdca"].keys())
print(len(train_keys), len(test_keys))

for i in train_keys:
  total_keys.append(i)

# for i in test_keys:
#   total_keys.append(i)

print(len(total_keys))
total_keys = shuffle(total_keys, random_state = 0)
keys_train, keys_test = train_test_split(total_keys, test_size=0.1, random_state=0)
print(len(keys_train), len(keys_test))

sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()

num_classes = 7

if num_classes == 7:
  # Original weight: weights = [1.0390098319422378, 4.199235235278951, 3.653153529987459, 2.9418323261480058, 1.9277996615169354, 1.835022318458948, 0.14040866052974108]
  weights = [2.0390098319422378, 5.199235235278951, 4.653153529987459, 2.9418323261480058, 1.9277996615169354, 1.835022318458948, 0.14040866052974108]
  weights = np.asarray(weights)

def generator_from_file(key_lst, num_classes, batch_size=1):
  # key_lst = list(h5file['gdca'].keys())
  random.shuffle(key_lst)
  i = 0

  while True:
      # TODO: different batch sizes with padding to max(L)
      # X_batch = []
      # y_batch = []
      # for j in range(batch_size):
      # index = random.randint(1, len(features)-1)
      if i == len(key_lst):
          random.shuffle(key_lst)
          i = 0

      key = key_lst[i]
      # print(key)

      if key in train_keys:
        X, y = get_datapoint(f_train, sequence_predictions, key, num_classes)
      else:
        X, y = get_datapoint(f_test, sequence_predictions, key, num_classes)

      batch_labels_dict = {}

      batch_labels_dict["out_dist"] = y

      i += 1

      yield X, batch_labels_dict

bs = 1
train_gen = generator_from_file(keys_train, num_classes = num_classes)
test_gen = generator_from_file(keys_test, num_classes = num_classes)

# model
model = actual_resnet(num_classes = num_classes)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/ares_1.h5', save_best_only=True, monitor='val_loss', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

sgd = keras.optimizers.SGD(learning_rate = 1e-3)
adam = keras.optimizers.Adam(learning_rate = 1e-2)
rms = keras.optimizers.RMSprop(learning_rate = 1e-7)
loss_fn = focal_loss()
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ['accuracy'])
with tf.device('/gpu:0'):
  history = model.fit_generator(train_gen, epochs = 100, steps_per_epoch = len(keys_train), verbose=1, shuffle = False, validation_data = test_gen, validation_steps = len(keys_test), callbacks = callbacks_list)
  
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
'''
2891 samples in train
###
Weights:[2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
Mis: [0.01857585 0.48648649 0.59133127 0.91813602 0.59436834 0.77357032  0.99131758]
Increase weight for 6
###
Weights: [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 1.8463084]
[0.28482972 0.38996139 0.40402477 0.81360202 0.96499239 0.99072643 0.90861081]
###
[2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 184.63084]
[0.94272446 0.95173745 0.9752322  0.99496222 0.99695586 0.78825348
 0.29269603]
###
[0.3942871, 0.3161621, 0.3942871, 0.4846191, 0.802002,  0.7897949, 0.68188477]
[0.76625387 1.         1.         1.         1.         1.
 0.00465449]
###

'''


