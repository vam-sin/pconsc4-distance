# author: Vamsi N.
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

# Libraries
import h5py
from preprocess_pcons import get_datapoint_align, get_datapoint
from alignment_process import _generate_features
from keras.utils import to_categorical 
from keras.models import load_model 
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
from tqdm import tqdm

# parameters
# Number of bins used for classification (Based on model_name)
num_classes = 37
bins = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20]
mids = [20.25, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75]

# Distance threshold to calculate all the other measures (8 or 15)
thres = 8.0

# The value n which is multiplied with L (Length of protein) to get the top n*L contacts
threshold_length = 1

# test data
test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')
key_lst = list(f_test["gdca"].keys())
pred_folder = '/home/vamsi/Internships/PConsC4_Distance/Repos/bioinfo-toolbox/trRosetta/predictions/'

# prob bins to dist
def distance_from_bins(pred, mids):
  # dist = mids[np.argmax(pred)]
  dist = 0.0
  for i in range(len(pred)):
    dist += pred[i] * mids[i]

  return dist

def percent_mismatches(y_true, y_pred):
  # print(y_pred.shape, y_true.shape)
  # y_pred = np.squeeze(y_pred, axis=0)
  # print(y_pred.shape)
  y_pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[0], num_classes))
  y_pred = np.argmax(y_pred, axis=1)
  # print(y_true.shape)
  # y_true = np.squeeze(y_true, axis=0)
  y_true = np.reshape(y_true, (y_true.shape[0]*y_true.shape[0]))
  # y_true = np.argmax(y_true, axis=1)
  # print(y_true.shape, y_pred.shape)
  cm = confusion_matrix(y_true, y_pred)
  import seaborn as sns
  import matplotlib.pyplot as plt

  sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', annot_kws={"fontsize":10})
  plt.show()
  return confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)

from keras.utils import to_categorical 

def WCCE(y_true, y_pred):
  weights = [1,1,1,1,0.1,0.1,0.1]
  bins = [4, 6, 8, 10, 12, 14] # 
  # mids = [0, 5, 7, 9, 11, 13, 15, 2]
  mids = [2, 5, 7, 9, 11, 13, 15]
  y_true = np.searchsorted(bins, y_true)
  y_pred = np.searchsorted(bins, y_pred)
  y_pred = to_categorical(y_pred, num_classes = 7)
  y_true = to_categorical(y_true, num_classes = 7)
  # print(y_true.shape, y_pred.shape)
  epsilon = 1e-5
  Lsq = y_pred.shape[0]*y_pred.shape[0]
  # y_pred = np.squeeze(y_pred, axis=0)
  # print(y_pred.shape)
  y_pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[0], 7))
  # print(y_true.shape)
  # y_true = np.squeeze(y_true, axis=0)
  # y_true = np.expand_dims(y_true, axis=0)
  # print(y_pred.shape, y_true.shape)
  # print(np.min(y_true))
  y_true = np.reshape(y_true, (y_true.shape[1]*y_true.shape[1], 7))

  y_pred /= np.sum(y_pred, axis=-1, keepdims=True) # clip to prevent NaN's and Inf's
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  loss = y_true * np.log(y_pred) * weights
  loss = -np.sum(loss, -1)
  loss = np.sum(loss, axis=0)

  loss /= Lsq
    
  return loss

prec = []

def mean_squared_error(y_true, y_pred):
    # y_true = np.squeeze(y_true, axis=0)
    # y_true = np.squeeze(y_true, axis=2)
    # y_pred = np.squeeze(y_pred, axis=0)
    # y_pred = np.squeeze(y_pred, axis=2)
    # print(y_pred - y_true)
    print(np.sum(np.abs(y_pred - y_true)))
    return np.sqrt(np.mean(np.square(y_pred - y_true)))

num_samples = 210
with tf.device('/cpu:0'):
  for sample in tqdm(range(1)): # sample = 15, 34 trial
    pred_file = pred_folder + key_lst[sample].replace('.hhE0', '') + '.npz'
    data = np.load(pred_file)
    y_pred = data["dist"]

    # distance calculation
    y_true_dist = f_test["dist"][key_lst[sample]][()]
    y_true_dist[y_true_dist > 20.0] = 20.25
    # print(y_pred.shape, y_true_dist.shape, len(bins))
    y_pred_ = np.zeros((y_true_dist.shape))
    for i in range(y_true_dist.shape[0]):
      for j in range(y_true_dist.shape[1]):
        y_pred_[i][j] = distance_from_bins(y_pred[i][j], mids)
    # print("Error: ", mean_squared_error(y_true_dist, y_pred_))
    L = len(y_true_dist)

    y_true_class = np.searchsorted(bins, y_true_dist)
    cm, cr = percent_mismatches(y_true_class, y_pred)
    for i in cm:
      print(i)
    print(cr)
    print("WCCE", WCCE(y_true_dist, y_pred_))
    # print(y_true_class)
    y_pred_ = np.zeros((y_true_dist.shape))
    for i in range(y_true_dist.shape[0]):
      for j in range(y_true_dist.shape[1]):
        y_pred_[i][j] = np.argmax(y_pred[i][j])
    y_pred_ += 1
    y_pred_[y_pred_ == 37] = 0
    # print(y_pred_) 

    # for i in range(L):
    #   for j in range(L):
    #     print(y_true_dist[i][j], y_pred_[i][j])

    # print(np.mean(abs_val))
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

    # check with ground truth
    # give all y_pred 1
    # for each i, j in y_pred, if y_true[i][j] < 8.0 then 1, else 0
    y_pred_final = np.ones((num_choose,))
    y_true_final = []

    for i in range(num_choose):
      ind1 = int(rem_sorted_pred_dist[i][0])
      ind2 = int(rem_sorted_pred_dist[i][1])
      if y_true_dist[ind1][ind2] < thres:
        y_true_final.append(1)
      else:
        y_true_final.append(0)

    cm = confusion_matrix(y_true_final, y_pred_final)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    print(cm, precision)
    try:
      prec.append(precision[1])
    except:
      prec.append(1.0)


  prec = np.asarray(prec)
  print("PPV: ", np.mean(prec))

'''
Tasks:
Calculate distance using Aditi’s method and rank the residue pairs in ascending order. - Done
Remove those pairs that have less than 5 residues between them. - Done
Choose the Top L. - Done
In these top L check those that have a ground truth distance value less than 8 angstroms. - Done

Results: PPV

37 classes 
trRosetta PPV:  0.7781857073910398 (Thres: 8.0)
trRosetta has four predictions: distance (d), omega, theta and phi.
The distance range (2 to 20 Å) is binned into 36 equally spaced segments, 0.5 Å each, plus one bin indicating that residues are not in contact.
First class is non contact and from the second one make the 36 demarcations from 2-20.
'''
