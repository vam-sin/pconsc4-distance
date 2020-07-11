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
num_filters = 64

def self_outer(x):
    outer_x = x[ :, :, None, :] * x[ :, None, :, :]
    return outer_x

def add_2D_conv(inp, filters, kernel_size, data_format="channels_last", padding="same", depthwise_initializer=init, pointwise_initializer=init, depthwise_regularizer=reg, 
        pointwise_regularizer=reg):
	x = Conv2D(num_filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(inp)
	x = Dropout(dropout)(x)
	x = act()(x)
	x = BatchNormalization()(x)

	x = Conv2D(num_filters, kernel_size, data_format=data_format, padding=padding, kernel_initializer=depthwise_initializer, kernel_regularizer=depthwise_regularizer)(inp)
	x = Dropout(dropout)(x)
	x = act()(x)
	x = BatchNormalization()(x)

	return x

def unet(num_classes):
	inp_2d = [Input(shape=(None, None, 1), name="gdca", dtype=K.floatx()), # gdca
                 Input(shape=(None, None, 1), name="mi_corr", dtype=K.floatx()), # mi_corr
                 Input(shape=(None, None, 1), name="nmi_corr", dtype=K.floatx()), # nmi_corr
                 Input(shape=(None, None, 1), name="cross_h", dtype=K.floatx())] # cross_h
                 
	inputs_seq = [Input(shape=(None, 22), dtype=K.floatx(), name="seq"), # sequence
	              Input(shape=(None, 23), dtype=K.floatx(), name="self_info"), # self-information
	              Input(shape=(None, 23), dtype=K.floatx(), name="part_entr")] # partial entropy

	ss_model = load_model('1d.h5')
	ss_model.trainable = False

	seq_feature_model = ss_model._layers_by_depth[5][0]
	#plot_model(seq_feature_model, "seq_feature_model.png")

	assert 'model' in seq_feature_model.name, seq_feature_model.name
	seq_feature_model.name = 'sequence_features'
	seq_feature_model.trainable = False
	for l in ss_model.layers:
	    l.trainable = False
	for l in seq_feature_model.layers:
	    l.trainable = False

	bottleneck_seq = seq_feature_model(inputs_seq)
	model_1D_outer = Lambda(self_outer)(bottleneck_seq)
	model_1D_outer = BatchNormalization()(model_1D_outer)

	#Downsampling
	unet = keras.layers.concatenate(inp_2d +  [model_1D_outer])
	unet = add_2D_conv(unet, num_filters, 1)
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
	        padding = "same", kernel_initializer = init, kernel_regularizer = reg, name="out_dist")(unet)

	model = Model(inputs = inp_2d + inputs_seq, outputs = output)
	print(model.summary())

	return model 

if __name__ == '__main__':
	model = unet(7)

# model ideas
'''
turn the add_2d_conv into residual blocks
turn the add_2d_conv into dense blocks
'''

