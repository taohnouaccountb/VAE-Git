from util import *

'''
# Model Description
## Conv2d (name_scope=_)
{'type': 'regu2d', 'filters': 100, 'k_size': 2, 'func': relu, 'name': 'encConv_0', 'strides': (1,1), 'reg': l2(scale=1.)},
{'type': 'pool', 'k_size': 2, 'strides': 2, 'name': 'decPool_0'},
{'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_0'},
{'type': 'trans2d', 'filters': 3, 'k_size': 1, 'scale': (1, 1), 'name': 'decConv_0'},
{'type': 'gan2d', 'filters': 3, 'k_size': 1, 'scale': 1, 'name': 'decConv_1'},

## Dense (name_scope=_, training=_)
{'type': 'dense', 'size': 500, 'reg': l2(scale=1.0),'func': relu,'name': 'hidden_layer_1'},
{'type': 'drop', 'rate': 0.5, 'name': 'dropout_3'},
'''

def encoder_bundle(x,training=None, name='encoder'):
    l2 = tf.contrib.layers.l2_regularizer
    relu = tf.nn.relu
    conv_layers = i_conv_layers(x, [
        {'type': 'regu2d', 'filters': 200, 'k_size': 2, 'func': relu, 'name': 'encConv_0', 'strides': (1,1), 'reg': l2(scale=1.)},
        {'type': 'regu2d', 'filters': 16, 'k_size': 2, 'func': relu, 'name': 'conv2d_1', 'strides': (2,2), 'reg': l2(scale=1.), },
    ])
    conv_layers = tf.cast(conv_layers, tf.uint8)
    return tf.identity(conv_layers, name=name)


def decoder_bundle(x,training=None, name='decoder'):
    x=tf.cast(x,tf.float32)
    l2 = tf.contrib.layers.l2_regularizer
    relu = tf.nn.relu
    conv_layers = i_conv_layers(x, [
        {'type': 'gan2d', 'filters': 200, 'k_size': 2, 'scale': 2, 'name': 'decConv_-1'},
        # {'type': 'trans2d', 'filters': 6, 'k_size': 2, 'scale': (1, 1), 'name': 'decConv_0'},
        # {'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_0'},
        # {'type': 'sub2d', 'k_size': 2, 'scale': 2, 'name': 'decSub_1'},
        # {'type': 'trans2d', 'filters': 3, 'k_size': 1, 'scale': (1, 1), 'name': 'decConv_1'},
        {'type': 'gan2d', 'filters': 3, 'k_size': 2, 'scale': 1, 'name': 'decConv_1'},
    ])
    conv_layers=tf.cast(conv_layers, tf.uint8)
    return tf.identity(conv_layers, name=name)
