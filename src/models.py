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

# Functions
def self_outer(x):
    outer_x = x[ :, :, None, :] * x[ :, None, :, :]
    return outer_x

###### trRosetta Model ######
def trBlock(inp, num_filters):
	x = Conv2D(num_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(inp)
	x = ELU()(x)
	x = BatchNormalization()(x)
	x = Dropout(0.1)(x)
	x = Conv2D(num_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = ELU()(x)
	x = BatchNormalization()(x)

	x = keras.layers.concatenate([x, inp])

	return x


def trRosetta(num_classes, num_filters):
	inp_2d = [Input(shape=(None, None, 1), name="gdca", dtype=K.floatx()), # gdca
                 Input(shape=(None, None, 1), name="mi_corr", dtype=K.floatx()), # mi_corr
                 Input(shape=(None, None, 1), name="nmi_corr", dtype=K.floatx()), # nmi_corr
                 Input(shape=(None, None, 1), name="cross_h", dtype=K.floatx())] # cross_h
                 
	seq = [Input(shape = (None, 128), dtype=K.floatx(), name = "seq_input")]
	model_1D_outer = Lambda(self_outer)(seq[0])
	model_1D_outer = BatchNormalization()(model_1D_outer)

	# #Downsampling
	x = keras.layers.concatenate(inp_2d +  [model_1D_outer])
	
	x = Conv2D(filters = num_filters, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = ELU()(x)

	for j in range(1):
		x = Block(x, num_filters)

	x = ELU()(x)

	output = Conv2D(num_classes, kernel_size = (1,1), padding = 'same', activation ="softmax", data_format = "channels_last", kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), name="out_dist")(x)
	# output = Reshape((-1, num_classes), name="out_dist")(x)
	# model = Model(inputs = inp_2d + inputs_seq, outputs = output)
	model = Model(inputs = inp_2d + seq, outputs = output)
	print(model.summary())

	return model 