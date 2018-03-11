from util import *

'''
# Model Description
## Conv2d (name_scope=_)
{'type': 'regu2d', 'filters': 100, 'k_size': 2, 'func': relu, 'name': 'encConv_0', 'reg': l2(scale=1.)},
{'type': 'pool', 'k_size': 2, 'strides': 2, 'name': 'decPool_0'},
{'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_0'},
{'type': 'trans2d', 'filters': 3, 'k_size': 1, 'scale': (1, 1), 'name': 'decConv_0'},

## Dense (name_scope=_, training=_)
{'type': 'dense', 'size': 500, 'reg': l2(scale=1.0),'func': relu,'name': 'hidden_layer_1'},
{'type': 'drop', 'rate': 0.5, 'name': 'dropout_3'},
'''

def encoder_bundle(x,training=None):
    l2 = tf.contrib.layers.l2_regularizer
    relu = tf.nn.relu
    conv_layers = i_conv_layers(x, [
        {'type': 'regu2d', 'filters': 100, 'k_size': 2, 'func': relu, 'name': 'encConv_0', 'reg': l2(scale=1.)},
        {'type': 'regu2d', 'filters': 200, 'k_size': 3, 'func': relu, 'name': 'conv2d_1', 'reg': l2(scale=1.), },
        {'type': 'pool', 'k_size': 2, 'strides': 2, 'name': 'decPool_0'},
        {'type': 'regu2d', 'filters': 200, 'k_size': 2, 'func': relu, 'name': 'encConv_0', 'reg': l2(scale=1.)},
        {'type': 'pool', 'k_size': 2, 'strides': 2, 'name': 'decPool_1'},
        {'type': 'regu2d', 'filters': 100, 'k_size': 5, 'func': relu, 'name': 'conv2d_2', 'reg': l2(scale=1.)},
    ])
    flat = tf.layers.flatten(conv_layers)
    dense_layers = i_regular_dense_layers(flat,[
        {'type': 'drop', 'rate': 0.5, 'name': 'dropout_3'},
        {'type': 'dense', 'size': 500, 'reg': l2(scale=1.0),'func': relu,'name': 'hidden_layer_1'},
        {'type': 'dense', 'size': 100, 'reg': l2(scale=1.0),'func': relu,'name': 'hidden_layer_2'},
    ],training=training)
    return dense_layers


def decoder_bundle(x,training=None):
    l2 = tf.contrib.layers.l2_regularizer
    relu = tf.nn.relu
    dense_layers = i_regular_dense_layers(x, [
        {'type': 'dense', 'size': 256, 'reg': l2(scale=1.0), 'func': relu, 'name': 'hidden_layer_2'},
    ], training=training)
    conv = tf.reshape(dense_layers, [-1,8,8,4])
    conv_layers = i_conv_layers(conv, [
        # {'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_0'},
        {'type': 'trans2d', 'filters': 6, 'k_size': 2, 'scale': (1, 1), 'name': 'decConv_0'},
        {'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_0'},
        {'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_1'},
        {'type': 'trans2d', 'filters': 3, 'k_size': 1, 'scale': (1, 1), 'name': 'decConv_1'},
    ])
    return conv_layers
