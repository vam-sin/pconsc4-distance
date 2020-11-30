import numpy as np 
import h5py 
from preprocess_pcons import get_datapoint
from alignment_process import _generate_features
from keras.models import load_model 
import keras.backend as K
from sklearn.metrics import classification_report, confusion_matrix

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

def distance_from_bins(pred, mids):
  dist = 0.0
  for i in range(len(pred)):
    dist += pred[i] * mids[i]

  return dist

bins = [4, 6, 8, 10, 12, 14]
mids = [2, 5, 7, 9, 11, 13, 15]
# 0-4 is the first class
weights = [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
weights = np.asarray(weights)

num_classes = 7

# Distance threshold to calculate all the other measures (8 or 15)
thres = 8.0

# The value n which is multiplied with L (Length of protein) to get the top n*L contacts
threshold_length = 1

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')
num_classes = 7

def generator_from_file(h5file, num_classes, batch_size=1):
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

      X, y = get_datapoint(h5file, key, num_classes)

      batch_labels_dict = {}
      batch_labels_dict["out_dist"] = y

      i += 1

      yield X, batch_labels_dict

test_gen = generator_from_file(f_test, num_classes = num_classes)

def PPV_Metric(y_pred, y_true):
  y_pred = K.squeeze(y_pred, axis=0)
  y_true = K.squeeze(y_true, axis=0)

  # # distance calculation
  # y_true_dist = f_test["dist"][key][()]
  y_true_dist = []
  for i in range(len(y_true)):
    arr = []
    for j in range(len(y_true)):
      arr.append(mids[np.argmax(y_true[i][j])])
    y_true_dist.append(arr)

  L = 0
  for i in y_true_dist:
    L += 1

  y_pred_dist = [] # i, j, dist
  unique_i_j = []

  for i in range(L):
    for j in range(L):
      if i <= j:
        temp = [i, j]
      else:
        temp = [j, i]
      if temp not in unique_i_j:
        dist = distance_from_bins(y_pred[i][j], mids)
        row = [i, j, dist]
        y_pred_dist.append(row)
        unique_i_j.append(temp) # this is done so that only one of (i,j) & (j, i) is included (the matrix is symmetric)

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
    prec = precision[1]
  except:
    prec = 1.0

  return prec

def mean_squared_error(y_true, y_pred):
    return K.sum(K.mean(K.square(y_pred - y_true), axis=-1))

# model = load_model('models/model_unet_7_559.h5')
key_lst = list(f_test['gdca'].keys())

key = key_lst[0]
# foldername = key.replace('.hhE0', '')
# align_fname = 'testset/testing/benchmarkset/' + foldername + '/alignment.a3m'
# feature_dict, b, c = _generate_features(align_fname)
# print(feature_dict)
X, y_true = get_datapoint(f_test, key, num_classes)
# X, y_true = get_datapoint_align(f_test, feature_dict, key, num_classes)
y_pred = model.predict(X)
print(y_true.shape)
# y_true = np.squeeze(y_true, axis=0)
# y_true = np.squeeze(y_true, axis=2)
# y_pred = y_true.copy()
# y_pred[0][1] = 10.0
print(y_pred.shape, y_true[0][1])

print(mean_squared_error(y_true, y_pred))

# # y_pred = K.argmax(y_pred, axis=-1) 
# y_pred = K.squeeze(y_pred, axis=0)
# y_pred = K.reshape(y_pred, (-1, num_classes))
# # y_true = K.argmax(y, axis=-1)
# y_true = K.squeeze(y, axis=0)
# y_true = K.reshape(y_true, (-1, num_classes))

# y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
# # clip to prevent NaN's and Inf's
# y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
# # calc
# loss = y_true * K.log(y_pred) * weights
# loss = -K.sum(loss, -1)

# print(y_true.shape, y_pred.shape)
# print(loss)

# loss = K.sum(loss, axis=0)
# print(loss)
# print(y_pred, y_true)

