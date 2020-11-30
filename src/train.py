# tasks
# Make your own custom weighted categorical cross entropy loss function

# libraries
import numpy as np 
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

def soft_dice_loss(epsilon=1e-6):

  def sdl(y_true, y_pred):
    axes = tuple(range(1, len(y_pred.shape)-1)) 
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    
    return 1 - K.mean((numerator + epsilon) / (denominator + epsilon))

  return sdl

def topKloss(num_classes = 7):
  def topK(y_true, y_pred):
    y_pred = K.squeeze(y_pred, axis=0)
    y_pred = K.reshape(y_pred, (-1, num_classes))
    y_true = K.squeeze(y_true, axis=0)
    y_true = K.reshape(y_true, (-1, num_classes))

    loss = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
    loss.update_state(y_true, y_pred)

    return loss.result()
  return topK

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

def binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight):

    TN = tf.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 0)
    TP = tf.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 1)

    FP = tf.logical_and(K.eval(y_true) == 0, K.eval(y_pred) == 1)
    FN = tf.logical_and(K.eval(y_true) == 1, K.eval(y_pred) == 0)

    # Converted as Keras Tensors
    TN = K.sum(K.variable(TN))
    FP = K.sum(K.variable(FP))

    specificity = TN / (TN + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    return 1.0 - (recall_weight*recall + spec_weight*specificity)

def custom_loss(recall_weight, spec_weight):

    def recall_spec_loss(y_true, y_pred):
        return binary_recall_specificity(y_true, y_pred, recall_weight, spec_weight)

    # Returns the (y_true, y_pred) loss function
    return recall_spec_loss

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
# functions
# functions
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
total_keys = shuffle(total_keys, random_state = 42)
keys_train, keys_test = train_test_split(total_keys, test_size=0.2, random_state=42)
print(len(keys_train), len(keys_test))

sequence_predictions = np.load('sequence_predictions.npy',allow_pickle='TRUE').item()

num_classes = 7

if num_classes == 7:
  # Original weight: 
  weights = [0.10390098319422378, 0.4199235235278951, 0.3653153529987459, 0.29418323261480058, 0.19277996615169354, 0.1835022318458948, 0.014040866052974108]
  # weights = [2.0390098319422378, 5.199235235278951, 4.653153529987459, 2.9418323261480058, 1.9277996615169354, 1.835022318458948, 0.14040866052974108]
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

bs = 1
train_gen = generator_from_file(keys_train, num_classes = num_classes)
test_gen = generator_from_file(keys_test, num_classes = num_classes)

# model


mcp_save = keras.callbacks.callbacks.ModelCheckpoint('models/rosetta_6.h5', save_best_only=True, monitor='val_loss', verbose=1)
reduce_lr = keras.callbacks.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
callbacks_list = [reduce_lr, mcp_save]

sgd = keras.optimizers.SGD(learning_rate = 1e-3)
opt = keras.optimizers.Adam(learning_rate = 1e-4)
rms = keras.optimizers.RMSprop(learning_rate = 1e-4)
loss = dice_loss()
model.compile(optimizer = rms, loss = "categorical_crossentropy", metrics = ['accuracy'])
with tf.device('/gpu:0'):
  history = model.fit_generator(train_gen, epochs = 100, steps_per_epoch = len(keys_train), verbose=1, shuffle = False, validation_data = test_gen, validation_steps = len(keys_test), callbacks = callbacks_list)
  
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

'''
2891 samples in train
###
Weights:[2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]
Mis: [0.01857585 0.48648649 0.59133127 0.91813602 0.59436834 0.77357032  0.99131758]
Increase weight for 6
###
Weights: [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 1.8463084]
[0.28482972 0.38996139 0.40402477 0.81360202 0.96499239 0.99072643 0.90861081]
###
[2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 184.63084]
[0.94272446 0.95173745 0.9752322  0.99496222 0.99695586 0.78825348
 0.29269603]
###
[0.3942871, 0.3161621, 0.3942871, 0.4846191, 0.802002,  0.7897949, 0.68188477]
[0.76625387 1.         1.         1.         1.         1.
 0.00465449]
###

'''


