# libraries
import keras
import keras.backend as K
from keras.regularizers import l2
from keras.layers import Activation, Dense, Dropout, Flatten, Reshape, Permute
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization, Reshape
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.regularizers import l2

# model definition function
# param
num_filters = 16

def self_outer(x):
    outer_x = x[ :, :, None, :] * x[ :, None, :, :]
    return outer_x

def DenseBlock(x, layers, filters):
	for i in range(layers):
		x = BatchNormalization(gamma_regularizer = l2(0.0001), beta_regularizer = l2(0.0001))(x)
		x = LeakyReLU(alpha = 0.05)(x)
		x = Conv2D(filters, kernel_size = (3, 3), padding = 'same', kernel_initializer = 'he_uniform', data_format = 'channels_last')(x)
		x = Dropout(0.2)(x)

	return x

def TransitionDown(x, filters):
	x = BatchNormalization(gamma_regularizer = l2(0.0001), beta_regularizer = l2(0.0001))(x)
	x = LeakyReLU(alpha = 0.05)(x)
	x = Conv2D(filters, kernel_size = (1, 1), padding = 'same', kernel_initializer = 'he_uniform')(x)
	x = Dropout(0.2)(x)
	x = MaxPooling2D(pool_size = (2,2), strides = (2,2), data_format = 'channels_last')(x)

	return x

def TransitionUp(x, filters):
	x = Conv2DTranspose(filters, kernel_size = (2,2), strides = (2,2), padding = 'same', data_format = 'channels_last')(x)

	return x

def fcdensenet103(num_classes):
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

	x = keras.layers.concatenate(inp_2d +  [model_1D_outer])
	
	x = Conv2D(filters = num_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	
	link1 = DenseBlock(x, 4, num_filters)
	c1 = keras.layers.concatenate([link1, x])
	x = TransitionDown(c1, num_filters)
	
	link2 = DenseBlock(x, 5, num_filters)
	c2 = keras.layers.concatenate([link2, x])
	x = TransitionDown(c2, num_filters)

	link3 = DenseBlock(x, 7, num_filters)
	c3 = keras.layers.concatenate([link3, x])
	x = TransitionDown(c3, num_filters)

	link4 = DenseBlock(x, 10, num_filters)
	c4 = keras.layers.concatenate([link4, x])
	x = TransitionDown(c4, num_filters)

	# link5 = DenseBlock(x, 12, 48)
	# c5 = keras.layers.concatenate([link5, x])
	# x = TransitionDown(c5, 48)

	# # middle
	x = DenseBlock(x, 15, num_filters)

	# # Upsampling
	# x = TransitionUp(x, num_filters)
	# x = keras.layers.concatenate([c5, x])

	x = DenseBlock(x, 12, num_filters)
	x = TransitionUp(x, num_filters)
	x = keras.layers.concatenate([c4, x])
	
	x = DenseBlock(x, 10, num_filters)
	x = TransitionUp(x, num_filters)
	x = keras.layers.concatenate([c3, x])

	x = DenseBlock(x, 7, num_filters)
	x = TransitionUp(x, num_filters)
	x = keras.layers.concatenate([c2, x])

	x = DenseBlock(x, 5, 48)
	x = TransitionUp(x, 48)
	x = keras.layers.concatenate([c1, x])

	x = DenseBlock(x, 4, num_filters)

	output = Conv2D(num_classes, kernel_size = (1,1), padding = 'same', activation ="softmax", data_format = "channels_last", kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), name="out_dist")(x)

	model = Model(inputs = inp_2d + seq, outputs = output)
	print(model.summary())

	return model 

if __name__ == '__main__':
	model = fcdensenet103(7)

'''PPV
One Dense Block: 0.3473552428276617
Two Dense Blocks: 
'''

