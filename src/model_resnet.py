# recreation of the TrRosetta model
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

def Block(inp):
	x = ELU()(inp)
	x = Conv2D(num_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = BatchNormalization()(x)
	x = Dropout(0.1)(x)
	x = Conv2D(num_filters, kernel_size = 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(x)
	x = BatchNormalization()(x)

	# x = keras.layers.concatenate([x, inp])

	return x


def resnet(num_classes):
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

	# Downsampling
	inp = keras.layers.concatenate(inp_2d +  [model_1D_outer])
	
	x = Conv2D(filters = num_filters, kernel_size = 1, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), data_format='channels_last')(inp)
	
	for j in range(20):
		x = Block(x)

	x = ELU()(x)

	output = Conv2D(num_classes, kernel_size = (1,1), padding = 'same', activation ="softmax", data_format = "channels_last", kernel_initializer = 'he_uniform', kernel_regularizer = l2(0.0001), name="out_dist")(x)

	model = Model(inputs = inp_2d + inputs_seq, outputs = output)
	print(model.summary())

	return model 

if __name__ == '__main__':
	model = resnet(7)


'''PPV
One Dense Block: 0.3473552428276617
Two Dense Blocks: 
'''

