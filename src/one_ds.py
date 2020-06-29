# libraries
import numpy as np 
import pandas as pd
import h5py
import pickle
import random

file_name = "../Datasets/PconsC4-data/data/test_plm-gdca-phy-ss-rsa-eff-ali-mi_new.h5"
f = h5py.File(file_name, 'r')

X = []
y = []

num_classes = 7

def one_ds_generator(f, num_classes):
	key_lst = list(f['gdca'].keys())
	random.shuffle(key_lst)
	p = 0

	while True:
		if p == len(key_lst):
			random.shuffle(key_lst)
			p = 0

		key = key_lst[p]

		rows = len(f['gdca'][key][()])
		# print(key, count+1)
		for i in range(rows):
			for j in range(rows):
				X_rows = []
				X_rows.append(f['gdca'][key][()][i][j])
				X_rows.append(f['cross_h'][key][()][i][j])
				X_rows.append(f['nmi_corr'][key][()][i][j])
				X_rows.append(f['mi_corr'][key][()][i][j])
				# print(X_rows)

				for k in f['seq'][key][()][i]:
					X_rows.append(k)
				for k in f['part_entr'][key][()][i]:
					X_rows.append(k)
				for k in f['self_info'][key][()][i]:
					X_rows.append(k)

				for k in f['seq'][key][()][j]:
					X_rows.append(k)
				for k in f['part_entr'][key][()][j]:
					X_rows.append(k)
				for k in f['self_info'][key][()][j]:
					X_rows.append(k)

				# print(len(X_rows))

				# y
				group = f['dist']
				dist = group[key][()][i][j]

				if num_classes == 7:
				    bins = [2, 5, 7, 9, 11, 13, 15]
				elif num_classes == 12:
				    bins = [2.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5]
				elif num_classes == 26:
				    bins = [2, 4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25]

				max_d = bins[len(bins)-1]

				dis_class = np.zeros(num_classes)
				if str(dist) == 'inf' or dist > max_d:
				    dis_class[num_classes-1] = 1.0

				for k in range(len(bins)-1):
				    if dist >= bins[k] and dist < bins[k+1]:
				        dis_class[k] = 1.0
				        break

				# for h in range(20):
				# 	X_rows.append(0.0)


				X = np.expand_dims(np.asarray(X_rows), axis=0)
				# print(X.shape)
				y = np.expand_dims(np.asarray(dis_class), axis=0)
				# print(y.shape)
				yield X, y


if __name__ == '__main__':

	X = np.asarray(X)
	y = np.asarray(y)
	print(X.shape, y.shape)

	# # Pickle
	filename = 'X_train.pickle'
	outfile = open(filename,'wb')
	pickle.dump(X,outfile)
	outfile.close()

	filename = 'y_train.pickle'
	outfile = open(filename,'wb')
	pickle.dump(y,outfile)
	outfile.close()




    