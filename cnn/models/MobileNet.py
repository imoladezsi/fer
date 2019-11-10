from keras.optimizers import *
from keras.models import Model
from keras.layers import *
from cnn.Constants import Constants
from cnn.models.ModelInterface import ModelInterface
'''
Keras MobileNet
https://github.com/arthurdouillard/keras-mobilenet/blob/master/mobilenet.py

It was used for experimentation, but it has no been included in the final models
'''


def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
    """Convolutional block having the Conv layer, Batch Normalization and ReLU activation"""
    channels = int(channels * alpha)

    x = Conv2D(channels,
               kernel_size=(3, 3),
               strides=strides,
               use_bias=False,
               padding='same',
               name='{}_conv'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn'.format(name))(x)
    x = Activation('relu', name='{}_act'.format(name))(x)
    return x


def get_dw_sep_block(tensor, channels, strides, alpha=1.0, name=''):
    """Depthwise separable conv: A Depthwise conv followed by a Pointwise conv."""
    channels = int(channels * alpha)

    # Depthwise
    x = DepthwiseConv2D(kernel_size=(3, 3),
                        strides=strides,
                        use_bias=False,
                        padding='same',
                        name='{}_dw'.format(name))(tensor)
    x = BatchNormalization(name='{}_bn1'.format(name))(x)
    x = Activation('relu', name='{}_act1'.format(name))(x)

    # Pointwise
    x = Conv2D(channels,
               kernel_size=(1, 1),
               strides=(1, 1),
               use_bias=False,
               padding='same',
               name='{}_pw'.format(name))(x)
    x = BatchNormalization(name='{}_bn2'.format(name))(x)
    x = Activation('relu', name='{}_act2'.format(name))(x)
    return x


class MobileNet(ModelInterface):
    """
    Actual MobileNet implementation conforming to the 'interface'
    """
    @staticmethod
    def create_model(width, height,depth, num_classes=7, alpha=1.0, include_top=True, weights=None):
        x_in = Input(shape=(width,height, depth))

        x = get_conv_block(x_in, 32, (2, 2), alpha=alpha, name='initial')

        layers = {
            (64, (1, 1)),
            (128, (2, 2)),
            (128, (1, 1)),
            (256, (2, 2)),
            (256, (1, 1)),
            (512, (2, 2)),
            *[(512, (1, 1)) for _ in range(5)],
            (1024, (2, 2)),
            (1024, (2, 2))
        }

        for i, (channels, strides) in enumerate(layers):
            x = get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))

        if include_top:
            x = GlobalAvgPool2D(name='global_avg')(x)
            x = Dense(num_classes, activation='softmax', name='softmax')(x)

        model = Model(inputs=x_in, outputs=x)

        opt = SGD(lr=Constants.INIT_LR)
        model.compile(loss="categorical_crossentropy", optimizer=opt,
                      metrics=["accuracy"])
        model.summary()

        return model
