# libraries
import keras
import keras.backend as K
from keras.regularizers import l2
from keras.layers import Activation
from keras.layers.core import Lambda
from keras.models import Model, load_model
from keras.utils import np_utils, plot_model
from keras.layers.merge import concatenate
from keras.layers import Input, Dropout, BatchNormalization
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
num_filters = 64

def add_2D_conv(x, filters, kernel_size, data_format="channels_last", padding="same", depthwise_initializer=init, pointwise_initializer=init, depthwise_regularizer=reg, 
        pointwise_regularizer=reg):
	x = Conv2D(num_filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(x)
	x = Dropout(dropout)(x)
	x = act()(x)
	x = BatchNormalization()(x)

	return x

def unet(num_classes):
	inp = Input(shape = (496, 496, 5))

	#Downsampling
	unet = add_2D_conv(inp, num_filters, 1)
	unet = add_2D_conv(unet, num_filters, 3)
	unet = add_2D_conv(unet, num_filters, 3)

	link1 = unet

	unet = MaxPooling2D(pool_size=(2, 2), data_format = "channels_last", padding='same')(unet)
	unet = add_2D_conv(unet, num_filters*2, 3)
	unet = add_2D_conv(unet, num_filters*2, 3)

	link2 = unet

	unet = MaxPooling2D(pool_size=(2, 2), data_format = "channels_last", padding='same')(unet)
	unet = add_2D_conv(unet, num_filters*4, 3)
	unet = add_2D_conv(unet, num_filters*4, 3)

	link3 = unet

	unet = MaxPooling2D(pool_size=(2, 2), data_format = "channels_last", padding='same')(unet)
	unet = add_2D_conv(unet, num_filters*8, 3)
	unet = add_2D_conv(unet, num_filters*8, 3)

	link4 = unet

	unet = MaxPooling2D(pool_size=(2, 2), data_format = "channels_last", padding='same')(unet)
	unet = add_2D_conv(unet, num_filters*16, 3)
	unet = add_2D_conv(unet, num_filters*16, 3)

	#Upsampling
	unet = UpSampling2D((2,2), data_format = "channels_last")(unet)
	unet = add_2D_conv(unet, num_filters*8, 2)

	unet = keras.layers.concatenate([unet, link4])

	unet = add_2D_conv(unet, num_filters*8, 3)
	unet = add_2D_conv(unet, num_filters*8, 3)

	unet = UpSampling2D((2,2), data_format = "channels_last")(unet)
	unet = add_2D_conv(unet, num_filters*4, 2)

	unet = keras.layers.concatenate([unet, link3])

	unet = add_2D_conv(unet, num_filters*4, 3)
	unet = add_2D_conv(unet, num_filters*4, 3)

	unet = UpSampling2D((2,2), data_format = "channels_last")(unet)
	unet = add_2D_conv(unet, num_filters*2, 2)

	unet = keras.layers.concatenate([unet, link2])

	unet = add_2D_conv(unet, num_filters*2, 3)
	unet = add_2D_conv(unet, num_filters*2, 3)

	unet = UpSampling2D((2,2), data_format = "channels_last")(unet)
	unet = add_2D_conv(unet, num_filters, 2)

	unet = keras.layers.concatenate([unet, link1])

	unet = add_2D_conv(unet, num_filters, 3)
	unet = add_2D_conv(unet, num_filters, 3)

	output = Conv2D(num_classes, 7, activation ="softmax", data_format = "channels_last", 
	        padding = "same", kernel_initializer = init, kernel_regularizer = reg)(unet)

	model = Model(inputs = inp, outputs = output)
	# print(model.summary())

	return model 

