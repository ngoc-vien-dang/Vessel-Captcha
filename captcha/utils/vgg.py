from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import plot_model
from keras.optimizers import Adam

# function for creating a vgg block

def vgg_block(layer_in, n_filters, n_conv):
  for _ in range(n_conv):
    layer_in = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    Dropout(0.1)(layer_in)
  layer_in = MaxPooling2D((2, 2), strides=(2, 2))(layer_in)
  return layer_in

def vgg_block1(layer_in, n_filters, n_conv):
  for _ in range(n_conv):
    layer_in = Conv2D(n_filters, (3, 3), padding='same', activation='relu')(layer_in)
    Dropout(0.1)(layer_in)
  #layer_in = MaxPooling2D((2,2), strides=(2,2))(layer_in)
  return layer_in

def get_vgg():
  # define model input
  visible = Input(shape=(32, 32, 1))
  # add vgg module
  layer = vgg_block(visible, 64, 2)
  # add vgg module
  layer = vgg_block(layer, 128, 2)
  # add vgg module
  layer = vgg_block1(layer, 256, 4)
  final_conv = Flatten()(layer)
  final_conv = Dense(128, activation='relu', kernel_initializer='he_uniform')(final_conv)
  final_conv = Dense(1, activation='sigmoid')(final_conv)
  # create model
  model = Model(inputs=visible, outputs=final_conv)
  model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy', metrics=['accuracy'])
  print('VGG-net compiled.')
  # summarize model
  model.summary()
  return model

