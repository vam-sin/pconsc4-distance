# author: Vamsi N.

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
import sys 
import tensorflow as tf

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

def preprocess(alignment_file):
  feature_dict, b, c = _generate_features(alignment_file)
  feat_lst = ['gdca', 'cross_h', 'nmi_corr', 'mi_corr']
  # key_lst = list(f['gdca'].keys())
  x_i_dict = {}
  for feat in feat_lst:
      if feat in ['sep']:
          x_i = np.array(range(L))
          x_i = np.abs(np.add.outer(x_i, -x_i))
      elif feat in ['gneff']:
          x_i = h5file[feat][key][()]
          x_i = log10(x_i)
          # x_i = np.outer(x_i, x_i)
      elif feat in ['plm_J']:
          x_i = h5file[feat][key][()]
          L = x_i.shape[0]
      elif feat in ['cross_h', 'mi_corr', 'nmi_corr', 'plm', 'gdca']:
          x_i = feature_dict[feat]
          L = x_i.shape[0]
          x_i = x_i[..., None]
          x_i = np.squeeze(x_i, axis = 3)
          x_i = np.squeeze(x_i, axis = 0)
          # print(x_i.shape)
      else:
          x_i = feature_dict[feat]
          L = x_i.shape[0]
      # x_i = pad(x_i, pad_even)
      # print(x_i.shape)
      x_i_dict[feat] = x_i[None, ...]

  seq = feature_dict["seq"]
  # seq = pad(seq, pad_even)
  x_i_dict["seq"] = seq

  part_entr = feature_dict["part_entr"]
  # part_entr = pad(part_entr, pad_even)
  x_i_dict["part_entr"] = part_entr

  self_info = feature_dict["self_info"]
  # self_info = pad(self_info, pad_even)
  x_i_dict["self_info"] = self_info

  return x_i_dict

# model predictions
def distance_from_bins(pred, mids):
  dist = 0.0
  for i in range(len(pred)):
    dist += pred[i] * mids[i]

  return dist

model = load_model(model_name)

# total arguments 
alignment_file = sys.argv[1]
output_file = sys.argv[2]

with tf.device('/cpu:0'):
  print("Calculating Features from Alignment File.")
  X = preprocess(alignment_file)
  print("Predicting Distance from Features.")
  y_pred = model.predict(X)
  print("Saving Output.")
  np.save(output_file, y_pred)
