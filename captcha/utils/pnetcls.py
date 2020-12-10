"""
File name: pnetcls.py
Author: ngocviendang
Date created: July 13, 2020

This file defines the pnetcls architecture.
"""

from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.merge import concatenate
from keras.layers import Dropout
from keras.utils import plot_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Flatten

# function for creating a conv block
def conv_block(layer_in, n_filters, n_conv, dilated_rate):
	# add convolutional layers
	for _ in range(n_conv):
		layer_in = Conv2D(n_filters, (3,3), padding='same', dilation_rate = dilated_rate, activation='relu')(layer_in)
	return layer_in

def get_pnetcls(patch_size):
	# define model input
	visible = Input(shape=(patch_size, patch_size, 1))
	# add block1
	layer1 = conv_block(visible, 64, 2, 1)
	# add block2
	layer2 = conv_block(layer1, 64, 2, 2)
	# add block3
	layer3 = conv_block(layer2, 64, 3, 4)
	# add block4
	layer4 = conv_block(layer3, 64, 3, 8)
	# add block5
	layer5 = conv_block(layer4, 64, 3, 16)
	# concatenate 5 blocks
	layer_out = concatenate([layer1, layer2, layer3, layer4, layer5], axis=-1)
	# add drop-out
	layer_out = Dropout(0.5)(layer_out)
	# add 1x1 conv layer
	layer_out = Conv2D(128, (1,1), padding='same', dilation_rate = 1, activation='relu')(layer_out)
	# add drop-out
	layer_out = Dropout(0.5)(layer_out)
	# add 1x1 conv layer
	layer_out = Conv2D(1, (1,1), padding='same', dilation_rate = 1, activation='relu')(layer_out)
	# flatten layer_out and dense 
	layer_out = Flatten()(layer_out)
	layer_out = Dense(128, activation='relu')(layer_out)
	layer_out = Dense(1, activation='sigmoid')(layer_out)
	# create model
	model = Model(inputs=visible, outputs=layer_out)
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
	print('pnetcls compiled.')
	# summarize model
	model.summary()
	return model
	
