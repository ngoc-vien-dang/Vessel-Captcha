#Reference: https://www.kaggle.com/meownoid/tiny-resnet-with-keras-99-314
from keras.layers import Input, Conv2D, Activation, BatchNormalization, GlobalAveragePooling2D, Dense, Dropout
from keras.layers.merge import add
from keras.activations import relu, softmax, sigmoid
from keras.models import Model
from keras import regularizers

def block(n_output, upscale=False):
    # n_output: number of feature maps in the block
    # upscale: should we use the 1x1 conv2d mapping for shortcut or not

    # keras functional api: return the function of type
    # Tensor -> Tensor
    def f(x):
        # H_l(x):
        # first pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # first convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1,
                   padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        # second pre-activation
        h = BatchNormalization()(x)
        h = Activation(relu)(h)
        # second convolution
        h = Conv2D(kernel_size=3, filters=n_output, strides=1,
                   padding='same', kernel_regularizer=regularizers.l2(0.01))(h)
        # f(x):
        if upscale:
            # 1x1 conv2d
            f = Conv2D(kernel_size=1, filters=n_output,
                       strides=1, padding='same')(x)
        else:
            # identity
            f = x
        # F_l(x) = f(x) + H_l(x):
        return add([f, h])
    return f

def get_resnet():
    # input tensor is the 32x32 grayscale image
    input_tensor = Input((32, 32, 1))
    # first conv2d with post-activation to transform the input data to some reasonable form
    x = Conv2D(kernel_size=3, filters=16, strides=1, padding='same',
           kernel_regularizer=regularizers.l2(0.01))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    # F_1
    x = block(16)(x)
    # F_2
    x = block(16)(x)
    # last activation of the entire network's output
    x = BatchNormalization()(x)
    x = Activation(relu)(x)
    # average pooling across the channels
    # 28x28x48 -> 1x48
    x = GlobalAveragePooling2D()(x)
    # dropout for more robust learning
    x = Dropout(0.1)(x)
    # last softmax layer
    x = Dense(units=1, kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation(sigmoid)(x)
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Tiny Resnet compiled.')
    # summarize model
    model.summary()
    return model 

