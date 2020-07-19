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
from keras.models import load_model 
from keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import random
from tqdm import tqdm

# parameters
# Number of bins used for classification (Based on model_name)
num_classes = 7

# Distance threshold to calculate all the other measures (8 or 15)
thres = 8.0

# The value n which is multiplied with L (Length of protein) to get the top n*L contacts
threshold_length = 1

# define bins and pretrained models
if num_classes == 7:
    model_name = 'models/model_unet_7_672.h5'
    bins = [4, 6, 8, 10, 12, 14]
    mids = [2, 5, 7, 9, 11, 13, 15]
    # 0-4 is the first class
    weights = [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
    weights = np.asarray(weights)
elif num_classes == 12:
    model_name = 'models/unet_2d_1d_12.h5'
    bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    mids = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
elif num_classes == 26:
    model_name = 'models/unet_2d_1d_26.h5'
    bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
    mids = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]

# test data
test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

def generator_from_file(h5file, num_classes, batch_size=1):
  key_lst = list(h5file['gdca'].keys())
  i = 0

  while True:
      if i == len(key_lst):
          i = 0

      key = key_lst[i]
      foldername = key.replace('.hhE0', '')
      align_fname = 'testset/testing/benchmarkset/' + foldername + '/alignment.a3m'
      feature_dict, b, c = _generate_features(align_fname)
      # print(feature_dict)
      # X, y = get_datapoint(h5file, key, num_classes)
      X, y = get_datapoint_align(h5file, feature_dict, key, num_classes)

      batch_labels_dict = {}

      batch_labels_dict["out_dist"] = y

      i += 1

      yield X, batch_labels_dict, key

test_gen = generator_from_file(f_test, num_classes = num_classes)

# model predictions
# custom loss
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

loss_fn = weighted_cce_3d(weights)
model = load_model(model_name, custom_objects = {'cce_3d': loss_fn})

prec = []

num_samples = 210

for sample in tqdm(range(num_samples)):
  X, y_true, key = next(test_gen)
  y_pred = model.predict(X)
  y_pred = np.squeeze(y_pred, axis=0)

  # distance calculation
  y_true_dist = f_test["dist"][key][()]

  L = len(y_true_dist)
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

7 classes 
Vanilla CCE: PPV:  0.6715637744355714
'''
