from tensorflow import keras
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, Dense, BatchNormalization, ReLU
from tensorflow.keras.layers import Add, Multiply, Lambda, concatenate, Dropout, GlobalAvgPool2D, Activation
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

def get_conv_block(tensor, channels, strides, alpha=1.0, name=''):
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

def CHANNEL_ATTENTION(tensor, ):
    def channel_attention(x):
        # TODO:检查一下是不是三个filter
        x = DepthwiseConv2D(3, use_bias=False, depthwise_initializer='ones',padding='same')(x)
        x = Conv2D(1, 3, use_bias=False, kernel_initializer='ones', padding='same')(x)
        return x
        #return K.expand_dims(Add()([x[:,:,:,0], x[:,:,:,1], x[:,:,:,2]]))
    return Lambda(channel_attention, name='channel_attention_init')(tensor)

def MOBV1(shape, num_classes, alpha=1.0, include_top=True, weights=None):
    x_in = Input(shape=shape)

    #if x_in.shape[-1] == 3:
    #    x = CHANNEL_ATTENTION(x_in)
    #else:
    #    x = x_in
    x = get_conv_block(x_in, 32, (2, 2), alpha=alpha, name='initial')
    layers = [
        (64, (1, 1)),
        (128, (2, 2)),
        (128, (1, 1)),
        (256, (2, 2)),
        (256, (1, 1)),
        (512, (2, 2)),
        *[(512, (1, 1)) for _ in range(5)],
        (1024, (2, 2)),
        (1024, (2, 2))
    ]

    for i, (channels, strides) in enumerate(layers):
        x = get_dw_sep_block(x, channels, strides, alpha=alpha, name='block{}'.format(i))

    if include_top:
        x = GlobalAvgPool2D(name='global_avg')(x)
        x = Dense(num_classes, activation='softmax', name='softmax')(x)

    model = Model(inputs=x_in, outputs=x)

    if weights is not None:
        model.load_weights(weights, by_name=True)

    return model

if __name__ == '__main__':
    model = MOBV2([256,256,3], 10)