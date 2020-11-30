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
import math
from keras.models import load_model 
from sklearn.metrics import classification_report, confusion_matrix
import keras.backend as K
from keras import metrics
import tensorflow as tf

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

ss_model = load_model('1d.h5')
print(ss_model.summary())
# ss_model.trainable = False
sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()

seq_feature_model = ss_model._layers_by_depth[5][0]

def generator_from_file(h5file, num_classes):
  key_lst = list(h5file['gdca'].keys())
  # random.shuffle(key_lst)
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

      X, y = get_datapoint(h5file, sequence_predictions, key, num_classes)
      # X_batch.append(X)
      # inputs_seq = [X["seq"], X["self_info"], X["part_entr"]]
      # bottleneck_seq = np.asarray(seq_feature_model.predict(inputs_seq))
      # print(bottleneck_seq)
      # X["seq_input"] = bottleneck_seq

      batch_labels_dict = {}

      batch_labels_dict["out_dist"] = y
      # y_batch.append(batch_labels_dict)

      i += 1

      yield X, batch_labels_dict, key

train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

test_file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f_test = h5py.File(test_file_name, 'r')

num_classes = 7
train_gen = generator_from_file(f_train, num_classes = num_classes)

pred_seq = {}

# for i in range(2891):
# 	X, y_true, key = next(train_gen)
# 	inputs_seq = [X["seq"], X["self_info"], X["part_entr"]]
# 	bottleneck_seq = np.asarray(seq_feature_model.predict(inputs_seq))
# 	print(bottleneck_seq.shape, i, key)
# 	pred_seq[key] = bottleneck_seq

# test_gen = generator_from_file(f_test, num_classes = num_classes)

# for i in range(210):
# 	X, y_true, key = next(test_gen)
# 	inputs_seq = [X["seq"], X["self_info"], X["part_entr"]]
# 	bottleneck_seq = np.asarray(seq_feature_model.predict(inputs_seq))
# 	print(bottleneck_seq.shape, i, key)
# 	pred_seq[key] = bottleneck_seq

# np.save('sequence_predictions.npy', pred_seq) 

weights = np.zeros((7))

for i in range(2891):
  X, y_true, key = next(train_gen)
  print(y_true["out_dist"].shape, i)
  perc = np.zeros((7))
  L = len(y_true["out_dist"][0])
  for i in range(L):
    for j in range(L):
      perc[np.argmax(y_true["out_dist"][0][i][j])] += 1

  perc /= (L*L)
  weights += perc

weights /= 2891

weights /= np.sum(weights)

print(weights)

'''
Frequency:
[0.09624548 0.02381386 0.02737361 0.03399242 0.05187261 0.05449525 0.71220678]
'''

# test_gen = generator_from_file(f_test, num_classes = num_classes)

# for i in range(210):
#   X, y_true, key = next(test_gen)
#   inputs_seq = [X["seq"], X["self_info"], X["part_entr"]]
#   bottleneck_seq = np.asarray(seq_feature_model.predict(inputs_seq))
#   print(bottleneck_seq.shape, i, key)
#   pred_seq[key] = bottleneck_seq

# np.save('sequence_predictions.npy', pred_seq) 