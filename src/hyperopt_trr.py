# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
import sys
import h5py
from preprocess_pcons import get_datapoint
from model_trRosetta import trRosetta
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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
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

# original
def weighted_cce_3d(weights): 
  # shape of both (1, None, None, num_classes)
  @tf.function
  def cce_3d(y_true, y_pred):
    L = 0.0
    # Lsq = y_pred.shape[1] * y_pred.shape[1]
    y_pred = K.squeeze(y_pred, axis=0)
    for i in y_pred:
      L += 1.0
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.reshape(y_true, (-1, num_classes))

    y_pred /= K.sum(y_pred, axis=-1, keepdims=True) # clip to prevent NaN's and Inf's
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    loss = y_true * K.log(y_pred) * weights
    loss = -K.sum(loss, -1)
    loss = K.sum(loss, axis=0)

    loss /= (L*L)
      
    return loss

  return cce_3d


X = []
y = []
X_val = []
y_val = []
space = {'a': hp.uniform('a', 0.0,1.0),
            'b': hp.uniform('b', 0.0,1.0),
            'c': hp.uniform('c', 0.0,1.0),
            'd': hp.uniform('d', 0.0,1.0),
            'e': hp.uniform('e', 0.0,1.0),
            'f': hp.uniform('f', 0.0,1.0),
            'g': hp.uniform('g', 0.0,1.0)
        }
# functions
# functions
train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

# y_true = f_test["dist"]['3RNVA.hhE0'][()]

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
total_keys = shuffle(total_keys, random_state = 42)
keys_train, keys_test = train_test_split(total_keys, test_size=0.2, random_state=42)
print(len(keys_train), len(keys_test))

sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()

num_classes = 7

if num_classes == 7:
  # Original weight: weights = [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
  weights = [1, 1, 1, 1, 0.1, 0.1, 0.1]
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

def generator_from_file2(h5file, num_classes, batch_size=1):
  key_lst = list(h5file['gdca'].keys())
  # random.shuffle(key_lst)
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

      X, y = get_datapoint(h5file, sequence_predictions, key, num_classes)

      batch_labels_dict = {}

      batch_labels_dict["out_dist"] = y

      i += 1

      yield X, batch_labels_dict, key

bs = 1
train_gen = generator_from_file(keys_train, num_classes = num_classes)
test_gen1 = generator_from_file(keys_test, num_classes = num_classes)
test_gen = generator_from_file2(f_test, num_classes = num_classes)

X, y_true, key = next(test_gen)

# model
# model = trRosetta(num_classes = num_classes)

# mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/rosetta_4.h5', save_best_only=True, monitor='val_loss', verbose=1)
# reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
# callbacks_list = [reduce_lr, mcp_save]

# sgd = keras.optimizers.SGD(learning_rate = 1e-3)
# opt = keras.optimizers.Adam(learning_rate = 1e-4)
# rms = keras.optimizers.RMSprop(learning_rate = 1e-4)
# loss = weighted_cce_3d(weights)
# topk = topKacc()

# model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])

def f_nn(params):
  model = trRosetta(num_classes = num_classes)

  mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/rosetta_4.h5', save_best_only=True, monitor='val_accuracy', verbose=1)
  reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
  callbacks_list = [reduce_lr, mcp_save]

  # sgd = keras.optimizers.SGD(learning_rate = 1e-3)
  opt = keras.optimizers.Adam(learning_rate = 1e-4)
  # rms = keras.optimizers.RMSprop(learning_rate = 1e-4)
  weights = [params['a'], params['b'], params['c'], params['d'], params['e'], params['f'], params['g']]
  loss = weighted_cce_3d(weights)
  # topk = topKacc()

  model.compile(optimizer = opt, loss = loss, metrics = ['accuracy'])

  history = model.fit_generator(train_gen, epochs = 10, steps_per_epoch = len(keys_train), verbose=1, shuffle = False, validation_data = test_gen1, validation_steps = len(keys_test), callbacks = callbacks_list)
  
  model = load_model('models/rosetta_4.h5', custom_objects={'cce_3d': loss})
  mids = [2, 5, 7, 9, 11, 13, 15]
  thres = 8.0

  # The value n which is multiplied with L (Length of protein) to get the top n*L contacts
  threshold_length = 1
  y_pred = model.predict(X)
  # print("Losses")
  # print("Weighted CCE: ", WCCE(y_true["out_dist"], y_pred, weights))
  # print("Confusion Matrix") 
  # cm, cr = percent_mismatches(y_true["out_dist"], y_pred)
  # print(cm)
  # print(cr)
  # print("MSE LOSS: ", mean_squared_error(y_true["out_dist"], y_pred))

  # distance calculation
  y_true_dist = f_test["dist"][key][()]
  # y_true_dist[y_true_dist > 15] = 15
  # print(y_true_dist.shape, y_pred.shape)

  # dim = int(math.sqrt(y_pred.shape[1]))
  # print(dim)

  # y_pred = np.reshape(y_pred, (1, dim, dim, num_classes))
  # print(y_pred.shape)

  y_pred = np.squeeze(y_pred, axis=0)
  # y_pred = np.squeeze(y_pred, axis=2)
  # print(y_pred.shape, y_pred)
  # print(y_pred.shape)
  y_pred = y_pred[np.ix_(range(y_true_dist.shape[0]), range(y_true_dist.shape[1]))]
  print(y_pred.shape)

  # percent_mismatches(y_true["out_dist"], y_pred)

  L = len(y_true_dist)
  y_pred_dist = [] # i, j, dist
  unique_i_j = []
  # err = mean_squared_error(y_true_dist, y_pred)
  # print("Error: ", err)

  for i in range(L):
    for j in range(L):
      if i <= j:
        temp = [i, j]
      else:
        temp = [j, i]
      if temp not in unique_i_j:
        # print(len(y_pred[i][j]))
        # dist = y_pred[i][j]
        # print(y_true_dist[i][j], dist)
        dist = np.dot(y_pred[i][j], mids) # distance_from_bins(y_pred[i][j], mids)
        # dist = distance_from_bins(y_pred[i][j], mids)
        # print(mids[np.argmax(y_pred[i][j])], dist, y_true_dist[i][j])
        # dist = mids[int(y_pred[i][j][0])]
        row = [i, j, dist]
        y_pred_dist.append(row)
        unique_i_j.append(temp) # this is done so that only one of (i,j) & (j, i) is included (the matrix is symmetric)

  y_pred_dist = np.asarray(y_pred_dist)
  # print(y_pred_dist)

  # sort ascending order of dist
  sorted_y_pred_dist = y_pred_dist[np.argsort(y_pred_dist[:, 2])]
  # print(sorted_y_pred_dist)

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
  # print(rem_sorted_pred_dist)

  # choose top L
  num_choose = threshold_length * L 
  chosen_y_pred = rem_sorted_pred_dist[0:num_choose,]

  # check with ground truth
  # give all y_pred 1
  # for each i, j in y_pred, if y_true[i][j] < 8.0 then 1, else 0
  y_pred_cm = y_pred
  y_pred = np.ones((num_choose,))
  y_true = []
  cm_pred = []
  cm_true = []
  for i in range(len(chosen_y_pred)):
    ind1 = int(chosen_y_pred[i][0])
    ind2 = int(chosen_y_pred[i][1])
    # print(chosen_y_pred[, y_true_dist[ind1][ind2])
    if y_true_dist[ind1][ind2] < thres:
      # print("Success", ind1, ind2, chosen_y_pred[i][2], y_true_dist[ind1][ind2])
      y_true.append(1)
    else:
      # print("Fail", ind1, ind2, chosen_y_pred[i][2], y_true_dist[ind1][ind2])
      y_true.append(0)

  # cm_true = np.searchsorted(cm_true, bins)
  # print(confusion_matrix(cm_true, cm_pred))

  cm = confusion_matrix(y_true, y_pred)
  precision = np.diag(cm) / np.sum(cm, axis = 0)
  print("Confusion Matrix:\n ", cm)
  print("Precision: ", precision)
  try:
    prec = precision[1]
  except:
    prec = 1.0
  acc = prec
  sys.stdout.flush() 
  
  return {'loss': -acc, 'status': STATUS_OK}

with tf.device('/gpu:0'):
  trials = Trials()
  best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
  print('best: ', best)
  # model = keras.models.load_model('models/rosetta_2.h5', custom_objects = {'cce_3d': loss})

  # history = model.fit_generator(train_gen, epochs = 100, steps_per_epoch = len(keys_train), verbose=1, shuffle = False, validation_data = test_gen, validation_steps = len(keys_test), callbacks = callbacks_list)
  
  # plt.plot(history.history['accuracy'])
  # plt.plot(history.history['val_accuracy'])
  # plt.title('model accuracy')
  # plt.ylabel('accuracy')
  # plt.xlabel('epoch')
  # plt.legend(['train', 'test'], loc='upper left')
  # plt.show()
  # # summarize history for loss
  # plt.plot(history.history['loss'])
  # plt.plot(history.history['val_loss'])
  # plt.title('model loss')
  # plt.ylabel('loss')
  # plt.xlabel('epoch')
  # plt.legend(['train', 'test'], loc='upper left')
  # plt.show()

'''
best:  {'a': 0.1929317781884383, 'b': 0.18794372886630434, 'c': 0.7968968753896802, 'd': 0.06816248913430835, 'e': 0.3145274731526676, 'f': 0.0001403478005401193, 'g': 0.9281697476380049}
'''


