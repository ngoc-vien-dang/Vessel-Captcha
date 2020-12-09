"""
File name: wnetseg.py
Author: ngocviendang
Date created: July 13, 2020
This file defines the 2D-WnetSeg architecture.
Reference https://github.com/prediction2020/unet-vessel-segmentation
"""

from keras.models import Model
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Input, UpSampling2D, concatenate, BatchNormalization

def conv_block(m, num_kernels, kernel_size, strides, padding, activation, dropout, data_format, bn):
    """
    Bulding block with convolutional layers for one level.
    """
    n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
                      data_format=data_format)(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(dropout)(n)
    n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
                      data_format=data_format)(n)
    n = BatchNormalization()(n) if bn else n
    return n

def up_concat_block(m, concat_channels, pool_size, concat_axis, data_format):
    """
    Bulding block with up-sampling and concatenation for one level in the first 2D-Unet.
    """
    n = UpSampling2D(size=pool_size, data_format=data_format)(m)
    n = concatenate([n, concat_channels], axis=concat_axis)
    return n

def up_concat_block2(m, concat_channels1, concat_channels2, pool_size, concat_axis, data_format):
    """
    Bulding block with up-sampling and concatenation for one level in the second 2D-Unet.
    """
    n = UpSampling2D(size=pool_size, data_format=data_format)(m)
    n = concatenate([n, concat_channels1,concat_channels2], axis=concat_axis)
    return n

def get_wnetseg(patch_size, num_channels, activation, final_activation, optimizer, learning_rate, dropout,
             loss_function, metrics=None,
             kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=3,
             data_format='channels_last', padding='same', bn=False):
    """
    Defines the architecture of the wnetseg. 
    """
    if metrics is None:
        metrics = ['accuracy']
    if num_kernels is None:
        num_kernels = [64, 128, 256, 512, 1024]

    # specify the input shape
    inputs = Input((patch_size, patch_size, num_channels))
    # The first U-net
    # DOWN-SAMPLING PART (left side of the first U-net)
    # layers on each level: convolution2d -> dropout -> convolution2d -> max-pooling
 
    # level 0
    conv_0_down_1 = conv_block(inputs, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_0_1 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_0_down_1)

    # level 1
    conv_1_down_1 = conv_block(pool_0_1, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_1_1 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_1_down_1)

    # level 2
    conv_2_down_1 = conv_block(pool_1_1, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_2_1 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_2_down_1)
    
    # level 3
    conv_4_1 = conv_block(pool_2_1, num_kernels[3], kernel_size, strides, padding, activation, dropout, data_format, bn)

    # UP-SAMPLING PART (right side of the first U-net)
    # layers on each level: upsampling2d -> concatenation with feature maps of corresponding level from down-sampling
  
    # level 2
    concat_2_1 = up_concat_block(conv_4_1, conv_2_down_1, pool_size, concat_axis, data_format)
    conv_2_up_1 = conv_block(concat_2_1, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
    # level 1
    concat_1_1 = up_concat_block(conv_2_up_1, conv_1_down_1, pool_size, concat_axis, data_format)
    conv_1_up_1 = conv_block(concat_1_1, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 0
    concat_0_1 = up_concat_block(conv_1_up_1, conv_0_down_1, pool_size, concat_axis, data_format)
    conv_0_up_1 = conv_block(concat_0_1, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
    final_conv_1 = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
                               data_format=data_format)(conv_0_up_1)
    
    # The second U-net
    # DOWN-SAMPLING PART (left side of the second U-net)
    # level 0
    conv_0_down_2 = conv_block(final_conv_1, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_0_2 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_0_down_2)

    # level 1
    conv_1_down_2 = conv_block(pool_0_2, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_1_2 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_1_down_2)

    # level 2
    conv_2_down_2 = conv_block(pool_1_2, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_2_2 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_2_down_2)

    # level 3
    conv_4_2 = conv_block(pool_2_2, num_kernels[3], kernel_size, strides, padding, activation, dropout, data_format, bn)

    # UP-SAMPLING PART (right side of the second U-net)
    
    # level 2
    concat_2_2 = up_concat_block2(conv_4_2,conv_2_down_1,conv_2_down_2, pool_size, concat_axis, data_format)
    conv_2_up_2 = conv_block(concat_2_2, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
    # level 1
    concat_1_2 = up_concat_block2(conv_2_up_2,conv_1_down_1,conv_1_down_2, pool_size, concat_axis, data_format)
    conv_1_up_2 = conv_block(concat_1_2, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
                           
    # level 0
    concat_0_2 = up_concat_block2(conv_1_up_2,conv_0_down_1,conv_0_down_2, pool_size, concat_axis, data_format)
    conv_0_up_2 = conv_block(concat_0_2, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
    final_conv_2 = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
                               data_format=data_format)(conv_0_up_2)
    

    # configure the learning process via the compile function
    model = Model(inputs=inputs, outputs=final_conv_2)
    model.compile(optimizer=optimizer(lr=learning_rate), loss=loss_function,
                  metrics=metrics)
    print('wnetseg compiled.')

    # print out model summary to console
    model.summary()

    return model
