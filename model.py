from util import *


def encoder_bundle(x):
    l2 = tf.contrib.layers.l2_regularizer
    relu = tf.nn.relu
    conv_layers = i_conv_layers(x, [
        {'type': 'regu2d', 'filters': 27, 'k_size': 2, 'func': relu, 'name': 'encConv_0', 'reg': l2(scale=1.)},
        {'type': 'pool', 'k_size': 2, 'strides': 2, 'name': 'decPool_0'},
        {'type': 'regu2d', 'filters': 54, 'k_size': 2, 'func': relu, 'name': 'conv2d_1', 'reg': l2(scale=1.), },
        {'type': 'pool', 'k_size': 2, 'strides': 2, 'name': 'decPool_1'},
        {'type': 'regu2d', 'filters': 6, 'k_size': 5, 'func': relu, 'name': 'conv2d_2', 'reg': l2(scale=1.)},
    ])
    return conv_layers


def decoder_bundle(x):
    conv_layers = i_conv_layers(x, [
        {'type': 'trans2d', 'filters': 54, 'k_size': 2, 'scale': (2,2), 'name': 'decConv_0'},
        {'type': 'trans2d', 'filters': 3, 'k_size': 2, 'scale': (2,2), 'name': 'decConv_0'},
    ])
    return conv_layers
