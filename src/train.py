# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
import h5py
from preprocess_pcons import get_datapoint
from model_unet import unet
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

def distance_from_bins(pred, mids):
  dist = 0.0
  for i in range(len(pred)):
    dist += pred[i] * mids[i]

  return dist

def thresh_loss(thres = 8.0, threshold_length = 1, bins = [4, 6, 8, 10, 12, 14], mids = [2, 5, 7, 9, 11, 13, 15]):
  
  @tf.function
  def tl(y_true, y_pred):
    y_pred = K.squeeze(y_pred, axis=0)

    # distance calculation
    y_true_dist = K.squeeze(y_true, axis=0)

    L = 0
    for i in y_true_dist:
      L += 1
    print(L, y_true_dist.shape)
    
    y_pred_dist = [] # i, j, dist

    for i in range(L):
      for j in range(L):
        dist = distance_from_bins(y_pred[i][j], mids)
        row = [i, j, dist]
        y_pred_dist.append(row)

    y_pred_dist = np.asarray(y_pred_dist)

    # sort ascending order of dist
    sorted_y_pred_dist = y_pred_dist[np.argsort(y_pred_dist[:, 2])]

    # Remove that have less than five residues between them
    del_rows = []
    for i in range(len(sorted_y_pred_dist)):
      if abs(sorted_y_pred_dist[i][0] - sorted_y_pred_dist[i][1]) < 6:
        del_rows.append(i)

    rem_sorted_pred_dist = []
    for i in range(len(sorted_y_pred_dist)):
      if i not in del_rows:
        rem_sorted_pred_dist.append(sorted_y_pred_dist[i])

    rem_sorted_pred_dist = np.asarray(rem_sorted_pred_dist)

    # choose top L
    num_choose = threshold_length * L 
    chosen_y_pred = rem_sorted_pred_dist[0:num_choose,]

    # check with ground truth
    # give all y_pred 1
    # for each i, j in y_pred, if y_true[i][j] < 8.0 then 1, else 0
    y_pred = np.ones((num_choose,))
    y_true = []

    for i in range(len(chosen_y_pred)):
      ind1 = int(chosen_y_pred[i][0])
      ind2 = int(chosen_y_pred[i][1])
      if y_true_dist[ind1][ind2] < thres:
        y_true.append(1)
      else:
        y_true.append(0)

    cm = confusion_matrix(y_true, y_pred)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print(cm, precision)
    try:
      loss = cm[0][1]/(np.sum(cm))
    except:
      loss = 0.0

    return loss 

  return tl 





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
model = unet(num_classes = num_classes)

mcp_save = keras.callbacks.callbacks.ModelCheckpoint('fcdensenet103.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

loss_fn = thresh_loss()
sgd = keras.optimizers.SGD(learning_rate = 1e-3)
adam = keras.optimizers.Adam(learning_rate = 1e-2)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
with tf.device('/cpu:0'):
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



