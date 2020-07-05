import numpy as np 
import h5py 
from preprocess_pcons import get_datapoint
from keras.models import load_model 
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

weights = [2.00393768, 8.01948742, 6.77744028, 5.19939732, 3.37958688, 3.08356088, 0.18463084]

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

model = load_model('models/unet_2d_1d_7.h5')
key_lst = list(f_test['gdca'].keys())

y_pred = []

X, y = get_datapoint(f_test, key_lst[0], num_classes)
y_pred = model.predict(X)

# y_pred = K.argmax(y_pred, axis=-1) 
y_pred = K.squeeze(y_pred, axis=0)
y_pred = K.reshape(y_pred, (-1, num_classes))
# y_true = K.argmax(y, axis=-1)
y_true = K.squeeze(y, axis=0)
y_true = K.reshape(y_true, (-1, num_classes))

y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
# clip to prevent NaN's and Inf's
y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
# calc
loss = y_true * K.log(y_pred) * weights
loss = -K.sum(loss, -1)

print(y_true.shape, y_pred.shape)
print(loss)

loss = K.sum(loss, axis=0)
print(loss)
# print(y_pred, y_true)

