# author: Vamsi N.
# GPU
from keras.utils import to_categorical 
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
from model_trRosetta import trRosetta
import math

# parameters
# Number of bins used for classification (Based on model_name)
num_classes = 7

# Distance threshold to calculate all the other measures (8 or 15)
thres = 8.0

# The value n which is multiplied with L (Length of protein) to get the top n*L contacts
threshold_length = 1

# define bins and pretrained models
if num_classes == 7:
    model_name = 'models/rosetta_90.h5' 
    bins = [4, 6, 8, 10, 12, 14] # 
    # mids = [0, 5, 7, 9, 11, 13, 15, 2]
    mids = [2, 5, 7, 9, 11, 13, 15]
    '''Class meanings
    0: 0-4
    1: 4-6
    2: 6-8
    3: 8-10
    4: 10-12
    5: 12-14
    6: 14+0.23015873015873015
    '''
    # 0-4 is the first class
    weights = [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
    weights = np.asarray(weights)
elif num_classes == 12:
    model_name = 'models/resnet.h5'
    bins = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    mids = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
elif num_classes == 26:
    model_name = 'resnet_gpu.h5'
    bins = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16]
    mids = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]
elif num_classes == 37:
    model_name = 'models/res_3.h5'
    bins = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11, 11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 15.5, 16, 16.5, 17, 17.5, 18, 18.5, 19, 19.5, 20]
    mids = [1, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75]

# test data
test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

def percent_mismatches(y_true, y_pred):
  y_pred = np.squeeze(y_pred, axis=0)
  # print(y_pred.shape)
  y_pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[0], num_classes))
  y_pred = np.argmax(y_pred, axis=1)
  print(y_true.shape)
  y_true = np.squeeze(y_true, axis=0)
  y_true = np.reshape(y_true, (y_true.shape[0]*y_true.shape[0], num_classes))
  y_true = np.argmax(y_true, axis=1)
  print(y_true, y_pred)
  cm = confusion_matrix(y_true, y_pred)
  import seaborn as sns
  import matplotlib.pyplot as plt

  # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', annot_kws={"fontsize":10})
  # plt.show()
  return confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)

sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()

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

test_gen = generator_from_file2(f_test, num_classes = num_classes)

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

def WCCE(y_true, y_pred, weights):
  epsilon = 1e-5
  Lsq = y_pred.shape[1]*y_pred.shape[1]
  y_pred = np.squeeze(y_pred, axis=0)
  # print(y_pred.shape)
  y_pred = np.reshape(y_pred, (y_pred.shape[0]*y_pred.shape[0], num_classes))
  # print(y_true.shape)
  y_true = np.squeeze(y_true, axis=0)
  y_true = np.reshape(y_true, (y_true.shape[0]*y_true.shape[0], num_classes))

  y_pred /= np.sum(y_pred, axis=-1, keepdims=True) # clip to prevent NaN's and Inf's
  y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
  loss = y_true * np.log(y_pred) * weights
  loss = -np.sum(loss, -1)
  loss = np.sum(loss, axis=0)

  loss /= Lsq
    
  return loss

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

def soft_dice_loss(epsilon=1e-6):

  def sdl(y_true, y_pred):
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean((numerator + epsilon) / (denominator + epsilon))

  return sdl

def tvl():
  def tversky_loss(y_true, y_pred):
      alpha = 0.5
      beta  = 0.5
      
      ones = K.ones(K.shape(y_true))
      p0 = y_pred      # proba that voxels are class i
      p1 = ones-y_pred # proba that voxels are not class i
      g0 = y_true
      g1 = ones-y_true
      
      num = K.sum(p0*g0, (0,1,2,3))
      den = num + alpha*K.sum(p0*g1,(0,1,2,3)) + beta*K.sum(p1*g0,(0,1,2,3))
      
      T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
      
      Ncl = K.cast(K.shape(y_true)[-1], 'float32')
      return Ncl-T
  return tversky_loss

def distance_from_bins(pred, mids):
  predsort = np.sort(pred)
  # print(predsort)
  pred = list(pred)
  # dist = (mids[pred.index(predsort[-1])] + mids[pred.index(predsort[-2])])/2
  # dist = mids[pred.index(predsort[-2])]
  # dist = 0.0
  # for i in range(len(pred)):
  #   dist += pred[i] * mids[i]

  return dist

def mean_squared_error(y_true, y_pred):
    # y_true = np.squeeze(y_true, axis=0)
    # y_true = np.squeeze(y_true, axis=2)
    # y_pred = np.squeeze(y_pred, axis=0)
    # y_pred = np.squeeze(y_pred, axis=2)
    # print(y_true, y_pred)
    return np.mean(np.abs(y_pred - y_true))

def dice_loss(eps=1e-6):
  def gen_dice(y_true, y_pred):
      """both tensors are [b, h, w, classes] and y_pred is in logit form"""

      # [b, h, w, classes]
      pred_tensor = tf.nn.softmax(y_pred)
      y_true_shape = tf.shape(y_true)

      # [b, h*w, classes]
      y_true = tf.reshape(y_true, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])
      y_pred = tf.reshape(pred_tensor, [-1, y_true_shape[1]*y_true_shape[2], y_true_shape[3]])

      # [b, classes]
      # count how many of each class are present in 
      # each image, if there are zero, then assign
      # them a fixed weight of eps
      counts = tf.reduce_sum(y_true, axis=1)
      weights = 1. / (counts ** 2)
      weights = tf.where(tf.math.is_finite(weights), weights, eps)

      multed = tf.reduce_sum(y_true * y_pred, axis=1)
      summed = tf.reduce_sum(y_true + y_pred, axis=1)

      # [b]
      numerators = tf.reduce_sum(weights*multed, axis=-1)
      denom = tf.reduce_sum(weights*summed, axis=-1)
      dices = 1. - 2. * numerators / denom
      dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
      return tf.reduce_mean(dices)

  return gen_dice

import keras
def topKacc(num_classes = 7):
  # @tf.function
  def topK(y_true, y_pred):
    y_pred = K.squeeze(y_pred, axis=0)
    # L = 0
    # for i in y_pred:
    #   L += 1
    # L = K.cast(L, dtype = 'int64')
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.reshape(y_true, (-1, num_classes))

    acc = keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

    return acc
  return topK

def rec(y_true, y_pred):
  return keras.metrics.Recall(y_true, y_pred)

top = topKacc()
loss_fn = weighted_cce_3d(weights)
model = load_model(model_name, custom_objects = {'cce_3d': loss_fn, 'recall': rec}, compile=False)
# model = trRosetta(num_classes = 7)
# model.load_weights(model_name)

prec = []

num_samples = 210
with tf.device('/cpu:0'):
  for sample in range(num_samples):
    print("##### Test Sample ", sample+1, " of 210 #####")
    X, y_true, key = next(test_gen)
    y_pred = model.predict(X)
    print("Losses")
    print("Weighted CCE: ", WCCE(y_true["out_dist"], y_pred, weights))
    print("Confusion Matrix") 
    cm, cr = percent_mismatches(y_true["out_dist"], y_pred)
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
      prec.append(precision[1])
    except:
      prec.append(1.0)
    print("Intermediate Mean Precision:", np.mean(prec))

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
Vanilla CCE: PPV:  0.5591865122940344
'''

'''rosetta_2
[[ 634    4    0    2    0    6    0]
 [   0  321   45   80    9   58    5]
 [   0   58  280  134   42  117   15]
 [   0   50  119  259   84  266   16]
 [   0   54  102  333  261  516   48]
 [   0   33   81  297  160  692   31]
 [   0  219  448 2117  775 6581 1032]]
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       646
           1       0.43      0.62      0.51       518
           2       0.26      0.43      0.33       646
           3       0.08      0.33      0.13       794
           4       0.20      0.20      0.20      1314
           5       0.08      0.53      0.15      1294
           6       0.90      0.09      0.17     11172

    accuracy                           0.21     16384
   macro avg       0.42      0.46      0.35     16384
weighted avg       0.70      0.21      0.22     16384
Precision:  [       nan 0.30952381]

Lower Fscore for 1,2,6
More Fscore for 3,4,5

rosetta_1
[[  633     2     0     0     0     0    11]
 [    0   256    39    25     6     0   192]
 [    0    30   209    24    22     0   361]
 [    0    22    58    40    48    13   613]
 [    0    29    29    30   143    16  1067]
 [    0     8    15    13    44    55  1159]
 [    0    16    20     8    19    26 11083]]
              precision    recall  f1-score   support

           0       1.00      0.98      0.99       646
           1       0.71      0.49      0.58       518
           2       0.56      0.32      0.41       646
           3       0.29      0.05      0.09       794
           4       0.51      0.11      0.18      1314
           5       0.50      0.04      0.08      1294
           6       0.77      0.99      0.86     11172

    accuracy                           0.76     16384
   macro avg       0.62      0.43      0.46     16384
weighted avg       0.70      0.76      0.69     16384
Precision:  [       nan 0.25396825]

'''
