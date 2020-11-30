# libraries
import keras
import keras.backend as K
from keras.regularizers import l2
import tensorflow as tf
from keras.layers import Activation
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.utils import np_utils, plot_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization, Reshape
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Conv2D, Conv1D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose

# model definition function
# param
dropout = 0.1
smooth = 1.
act = ELU
init = "he_normal"
reg_strength = float(10**-12)
reg = l2(reg_strength)
num_filters = 32

def self_outer(x):
    outer_x = x[ :, :, None, :] * x[ :, None, :, :]
    return outer_x

def actual_resnet(num_classes):
	inp_2d = [Input(shape=(None, None, 1), name="gdca", dtype=K.floatx()), # gdca
                 Input(shape=(None, None, 1), name="mi_corr", dtype=K.floatx()), # mi_corr
                 Input(shape=(None, None, 1), name="nmi_corr", dtype=K.floatx()), # nmi_corr
                 Input(shape=(None, None, 1), name="cross_h", dtype=K.floatx())] # cross_h
                 
	# inputs_seq = [Input(shape=(None, 22), dtype=K.floatx(), name="seq"), # sequence
	#               Input(shape=(None, 23), dtype=K.floatx(), name="self_info"), # self-information
	#               Input(shape=(None, 23), dtype=K.floatx(), name="part_entr")] # partial entropy

	# ss_model = load_model('1d.h5')
	# ss_model.trainable = False

	# seq_feature_model = ss_model._layers_by_depth[5][0]
	# #plot_model(seq_feature_model, "seq_feature_model.png")

	# assert 'model' in seq_feature_model.name, seq_feature_model.name
	# seq_feature_model.name = 'sequence_features'
	# seq_feature_model.trainable = False
	# for l in ss_model.layers:
	#     l.trainable = False
	# for l in seq_feature_model.layers:
	#     l.trainable = False

	# bottleneck_seq = seq_feature_model(inputs_seq)
	seq = [Input(shape = (None, 128), dtype=K.floatx(), name = "seq_input")]
	model_1D_outer = Lambda(self_outer)(seq[0])
	model_1D_outer = BatchNormalization()(model_1D_outer)

	# #Downsampling
	inp = keras.layers.concatenate(inp_2d +  [model_1D_outer])

	x1 = Conv2D(filters = num_filters, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(inp)

	# Res 2
	x = Conv2D(filters = num_filters, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x1)
	x = Conv2D(filters = num_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = Conv2D(filters = num_filters, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x2 = keras.layers.concatenate([x, x1])

	# Res 3
	x = Conv2D(filters = num_filters*2, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x2)
	x = Conv2D(filters = num_filters*2, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = Conv2D(filters = num_filters*2, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x3 = keras.layers.concatenate([x, x2])

	# Res 4
	x = Conv2D(filters = num_filters*4, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x3)
	x = Conv2D(filters = num_filters*4, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = Conv2D(filters = num_filters*4, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x4 = keras.layers.concatenate([x, x3])

	# Res 4
	x = Conv2D(filters = num_filters*4, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x4)
	x = Conv2D(filters = num_filters*4, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = Conv2D(filters = num_filters*4, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x5 = keras.layers.concatenate([x, x4])

	output = Conv2D(num_classes, 1, activation ="softmax", data_format = "channels_last", 
	        padding = "same", kernel_initializer = init, kernel_regularizer = reg, name="out_dist")(x5)

	model = Model(inputs = inp_2d + seq, outputs = output)
	print(model.summary())

	return model 

if __name__ == '__main__':
	model = actual_resnet(7)

# model ideas
'''
turn the add_2d_conv into residual blocks
turn the add_2d_conv into dense blocks
'''

