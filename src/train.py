# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
import h5py
from preprocess_pcons import get_datapoint
from model import unet
import pickle
import random
import keras
from keras.models import load_model 
from sklearn.metrics import classification_report, confusion_matrix
import keras.backend as K

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

from keras import backend as K


def weighted_cce_3d(weights): 
  # shape of both (1, None, None, num_classes)
  @tf.function
  def cce_3d(y_true, y_pred):
    loss = 0.0
    y_pred = K.squeeze(y_pred, axis=0)
    y_true = K.squeeze(y_true, axis=0)
    shape = y_pred.shape
    # print(shape)
    L = 0
    for i in y_pred:
      L += 1
    # print(L)
    for i in range(L):
      for j in range(L):
        temp_true = y_true[i][j]
        temp_pred = y_pred[i][j]

        temp_pred /= K.sum(temp_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        temp_pred = K.clip(temp_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        temp_loss = temp_true * K.log(temp_pred) * weights
        temp_loss = -K.sum(temp_loss, -1)

        # print(temp_pred.shape, temp_true.shape)

        loss += temp_loss
        
    return loss

  return cce_3d




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

      # new_w = []
      # for p in range(y.shape[1]):
      #   for q in range(y.shape[1]):
      #     new_w.append(weights)

      #   # new_w.append(temp)

      # new_w = np.asarray(new_w)
      # new_w_shape = new_w.shape
      # new_w = np.reshape(new_w, (1, new_w_shape[0]*new_w.shape[1]))
      # # print(new_w.shape)

      # curr_y_shape = y.shape 
      # y = np.reshape(y, (curr_y_shape[1] * curr_y_shape[2], num_classes))
      batch_labels_dict["out_dist"] = y

      # print(y.shape, new_w.shape)

      i += 1

      yield X, batch_labels_dict

train_gen = generator_from_file(f_train, num_classes = num_classes)
test_gen = generator_from_file(f_test, num_classes = num_classes)

# model
model = unet(num_classes = num_classes)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('unet.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

loss_fn = weighted_cce_3d(weights = weights)
opt = keras.optimizers.SGD(learning_rate = 0.1)
model.compile(optimizer = opt, loss = loss_fn, metrics = ['accuracy'])
# with tf.device('/cpu:0'):
model.fit_generator(train_gen, epochs = 10, steps_per_epoch = 290, verbose=1, validation_data = test_gen, validation_steps = 210, callbacks = callbacks_list)

# PPV Evaludation 
# model = load_model('models/unet_2d_1d_26.h5')
# key_lst = list(f_test['gdca'].keys())
# y_true = []
# y_pred = []
# for i in range(210):
#   print(i+1)
#   X, y = get_datapoint(f_test, key_lst[i], num_classes)
#   y_pred.append(model.predict(X))
#   y_true.append(y)

# one_y_true = []
# one_y_pred = []

# for i in y_true:
#   # ith complex (None, None, num_classes)
#   # temp = np.asarray(i)
#   # print(temp.shape)
#   for j in i:
#     for k in j:
#       for p in k:
#         # num_classes
#         one_y_true.append(np.argmax(p))

# # temp1 = np.asarray(y_pred)
# # temp2 = np.asarray(y_true)
# # print(temp1.shape, temp2.shape)
# for i in y_pred:
#   # ith complex (None, None, num_classes)
#   # temp = np.asarray(i)
#   # print(temp.shape)
#   for j in i:
#     for k in j:
#       for p in k:
#         # print(p)

#         one_y_pred.append(np.argmax(p))


# print(len(one_y_pred), len(one_y_true))

# confusion_matrix = confusion_matrix(one_y_true, one_y_pred)

# FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
# FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
# TP = np.diag(confusion_matrix)
# TN = confusion_matrix.sum() - (FP + FN + TP)

# ACC = (TP+TN)/(TP+FP+FN+TN)
# PPV = TP/(TP+FP)

# print("Accuracy: ", ACC)
# print("PPV: ", PPV)

'''
num_classes = 7
Accuracy:  [0.99791577 0.98941037 0.98323674 0.97221216 0.96044709 0.95563355
 0.89367793]
PPV:  [0.98287278 0.75014834 0.62556969 0.42357862 0.52627918 0.50621767
 0.88414448]

num_classes = 12
Accuracy:  [0.99314271 0.99034338 0.99011985 0.98775791 0.98866754 0.98210719
 0.97598051 0.97912268 0.97710804 0.97602216 0.97366097 0.86859402]
PPV:  [0.97086422 0.63947233 0.44432123 0.42859469 0.13835341 0.34136207
 0.33094466 0.17966295 0.2859744  0.2692568  0.18350515 0.85527541]

num_classes = 26
Accuracy:  [0.99784002 0.9986482  0.99677071 0.99398201 0.99231159 0.99421924
 0.99485658 0.99200519 0.99425611 0.99480463 0.99396533 0.99207605
 0.98921521 0.98924368 0.98633832 0.98888448 0.9900747  0.98944288
 0.98803498 0.98801565 0.98786755 0.98719823 0.98649811 0.98540436
 0.9857864  0.8372842 ]
PPV:  [0.98327405 0.26422764 0.35851884 0.50287211 0.42459293 0.34808997
 0.20822712 0.35505697 0.15891505 0.1023622  0.12406217 0.08823246
 0.29009821 0.13604011 0.27299517 0.13679245 0.08991955 0.11254883
 0.24466154 0.13680602 0.16071429 0.15661765 0.12569832 0.31334832
 0.30307773 0.81865921]
'''






