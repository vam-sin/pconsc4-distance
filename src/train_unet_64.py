# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
import h5py
from split_preprocess_pcons import get_datapoint
from model_resnet import resnet
import pickle
import random
import keras
import math
from keras.models import load_model 
from sklearn.metrics import classification_report, confusion_matrix
import keras.backend as K
from keras import metrics
import matplotlib.pyplot as plt
import tensorflow as tf
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

def mean_squared_error(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.squeeze(y_true, axis=2)
    y_pred = K.squeeze(y_pred, axis=0)
    y_pred = K.squeeze(y_pred, axis=2)
    # y_pred[y_pred > 8] = 0
    # y_true[y_true > 8] = 0 
    return K.sum(K.abs(y_pred - y_true))

# functions
ns = 64 # new shape
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
# total_keys = shuffle(total_keys, random_state = 42)
keys_train, keys_test = train_test_split(total_keys, test_size=0.0001, random_state=42)
print(len(keys_train), len(keys_test))

train_steps = 0
for i in range(len(keys_train)):
  val = f_train["dist"][keys_train[i]][()].shape[0]-ns
  train_steps += (val*val)

test_steps = 0
for i in range(len(keys_test)):
  val = f_train["dist"][keys_test[i]][()].shape[0]-ns
  test_steps += (val*val)

print("Steps: ", train_steps, test_steps)

sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()

num_classes = 7

if num_classes == 7:
  # Original weight: weights = [1.0390098319422378, 4.199235235278951, 3.653153529987459, 2.9418323261480058, 1.9277996615169354, 1.835022318458948, 0.14040866052974108]
  weights = [2.0390098319422378, 5.199235235278951, 4.653153529987459, 2.9418323261480058, 1.9277996615169354, 1.835022318458948, 0.14040866052974108]
  weights = np.asarray(weights)

def generator_from_file(key_lst, num_classes, batch_size=1):
  # key_lst = list(h5file['gdca'].keys())
  random.shuffle(key_lst)
  k = 0

  while True:
      # TODO: different batch sizes with padding to max(L)
      # X_batch = []
      # y_batch = []
      # for j in range(batch_size):
      # index = random.randint(1, len(features)-1)
      if k == len(key_lst):
          random.shuffle(key_lst)
          k = 0

      key = key_lst[k]
      k += 1
      # print(key)

      if key in train_keys:
        X, y = get_datapoint(f_train, sequence_predictions, key, num_classes)
      else:
        X, y = get_datapoint(f_test, sequence_predictions, key, num_classes)
      # print(y.shape)
      for i in range(y.shape[0]-ns):
        for j in range(y.shape[1]-ns):
            x_ns = {}
            
            dist = y[np.ix_(range(i, i+ns), range(j, j+ns))]
            x_ns["dist"] = np.expand_dims(dist, axis=0)
            
            gdca = X["gdca"][np.ix_(range(i, i+ns), range(j, j+ns))]
            gdca = np.expand_dims(gdca, axis=0)
            gdca = np.expand_dims(gdca, axis=3)
            x_ns["gdca"] = gdca
            
            cross_h = X["cross_h"][np.ix_(range(i, i+ns), range(j, j+ns))]
            cross_h = np.expand_dims(cross_h, axis=0)
            cross_h = np.expand_dims(cross_h, axis=3)
            x_ns["cross_h"] = cross_h

            mi_corr = X["mi_corr"][np.ix_(range(i, i+ns), range(j, j+ns))]
            mi_corr = np.expand_dims(mi_corr, axis=0)
            mi_corr = np.expand_dims(mi_corr, axis=3)
            x_ns["mi_corr"] = mi_corr

            nmi_corr = X["nmi_corr"][np.ix_(range(i, i+ns), range(j, j+ns))]
            nmi_corr = np.expand_dims(nmi_corr, axis=0)
            nmi_corr = np.expand_dims(nmi_corr, axis=3)
            x_ns["nmi_corr"] = nmi_corr
            
            seq = np.squeeze(X["seq_input"], axis=0)
            seq_input = seq[np.ix_(range(i, i+ns))]
            x_ns["seq_input"] = np.expand_dims(seq_input, axis=0)
            # print(x_ns["seq_input"].shape)

            batch_labels_dict = {}

            batch_labels_dict["out_dist"] = x_ns["dist"]

            yield x_ns, batch_labels_dict

bs = 1
train_gen = generator_from_file(keys_train, num_classes = num_classes)
test_gen = generator_from_file(keys_test, num_classes = num_classes)

# model
model = resnet(num_classes = num_classes)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/res_64.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=10, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

sgd = keras.optimizers.SGD(learning_rate = 1e-3)
adam = keras.optimizers.Adam(learning_rate = 1e-4)
# loss_fn = weighted_cce_3d(weights)
model.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
with tf.device('/gpu:0'):
  history = model.fit_generator(train_gen, epochs = 100, steps_per_epoch = 10000, verbose=1, shuffle = False, validation_data = test_gen, validation_steps = test_steps, callbacks = callbacks_list)
  
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
# model.compile(optimizer = "nadam", loss = mean_squared_error, metrics = ['accuracy'])
# with tf.device('/gpu:0'):
#   model.fit_generator(train_gen, epochs = 1000, steps_per_epoch = math.ceil(2891/bs), verbose=1, shuffle = False, validation_data = test_gen, validation_steps = 210, workers = 0, callbacks = callbacks_list)


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


