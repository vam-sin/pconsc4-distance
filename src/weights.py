import h5py
from sklearn.utils import class_weight
from preprocess_pcons import get_datapoint
import numpy as np

train_file_name = "../Datasets/PconsC4-data/data/training_pdbcull_170914_A_before160501.h5"
f_train = h5py.File(train_file_name, 'r')

num_classes = 7

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

key_lst = list(f_train['gdca'].keys())
y_true = []

for i in range(2891):
  print(i+1)
  X, y = get_datapoint(f_train, key_lst[i], num_classes)
  y_true.append(y)

one_y_true = []

for i in y_true:
  for j in i:
    for k in j:
      for p in k:
        # num_classes
        one_y_true.append(np.argmax(p))

one_y_true = np.asarray(one_y_true)
class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(one_y_true),
                                                 one_y_true)

print(class_weights)

'''
7: [2.00393768 8.01948742 6.77744028 5.19939732 3.37958688 3.08356088 0.18463084]

'''

