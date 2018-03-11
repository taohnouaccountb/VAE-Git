import numpy as np
import tensorflow as tf
from functools import reduce


# Compute accuracy
def get_eye_ratio(conf_matrix):
    eye_matrix = np.eye(conf_matrix.shape[0], conf_matrix.shape[1])
    return (conf_matrix * eye_matrix).sum() / conf_matrix.sum()


# Shuffle the data instances
def random_shuffle(data, label):
    assert data.shape[0] == label.shape[0]
    size = data.shape[0]
    s = np.random.permutation(size)
    return data[s], label[s]


# Compute PSNR of two tensors
def PSNR(x, output):
    def log10(x):
        tf.cast(x,tf.float32)
        return tf.log(x) / tf.log(10.0)
    delta = x-output
    MSE = tf.reduce_mean(tf.square(delta))  # MSE
    PSNR = 20 * log10(255.0) - 10 * log10(MSE)
    return PSNR


def soft_add_attribute_to_dict(i, name, attr):
    if name not in i:
        i[name] = attr
    return i


# Layer auto generator tools
def i_regular_dense_layers(input_tensor,
                           layers_meta,
                           name_scope='layers_block',
                           training=None):
    print('DENSE_LAYER_STRUCTURE', [i['size'] if i['type'] == 'dense' else 0 for i in layers_meta])
    with tf.name_scope(name_scope) as scope:
        layers = [input_tensor, ]
        for i in layers_meta:
            if i['type'] == 'dense':
                cur_layer = tf.layers.Dense(i['size'],
                                            kernel_regularizer=i['reg'],
                                            bias_regularizer=i['reg'],
                                            activation=i['func'],
                                            name=i['name'])
                layers.append(cur_layer)
            elif i['type'] == 'drop':
                if training is None:
                    raise Exception("Dropout layer MUST have a training label argument")
                cur_layer = tf.layers.Dropout(i['rate'], name=i['name'])
                layers.append(cur_layer)
            else:
                raise Exception("Wrong layer type in i_regular_dense_layers")
        layers_group = layers[0]
        for i in layers[1:]:
            if isinstance(i, tf.layers.Dropout):
                layers_group = i(layers_group, training=training)
            else:
                layers_group = i(layers_group)
        return layers_group


def i_conv_layers(input_tensor, layers_meta, name_scope='conv_block'):
    class SubPixelUpScaling:
        """ [Sub-Pixel Convolution](https://arxiv.org/abs/1609.05158) """

        # Filters will be same the input layer
        def __init__(self, scale, k_size, strides=(1, 1), activation=tf.nn.relu, regularizer=tf.nn.l2_normalize,
                     name='inner_layer'):
            self.scale = scale
            self.strides = strides
            self.activation = activation
            self.regularizer = regularizer
            self.name = name
            self.k_size = k_size

        def __call__(self, input):
            n, w, h, c = input.get_shape().as_list()
            x = tf.layers.conv2d(input, c * self.scale ** 2, kernel_size=self.k_size,
                                 strides=self.strides,
                                 activation=self.activation,
                                 padding='same',
                                 name=self.name,
                                 bias_regularizer=self.regularizer,
                                 kernel_regularizer=self.regularizer)
            output = tf.depth_to_space(x, self.scale)
            return output

    class ProgressiveGanUpsampling:
        """ similar to the upsampling used in [ProgressiveGAN](https://arxiv.org/pdf/1710.10196.pdf) """

        def __init__(self, scale, filters, k_size, strides=(1, 1), activation=tf.nn.relu,
                     regularizer=tf.nn.l2_normalize,
                     name='inner_layer'):
            self.scale = scale
            self.strides = strides
            self.activation = activation
            self.regularizer = regularizer
            self.name = name
            self.filters = filters
            self.k_size = k_size

        def __call__(self, input):
            n, w, h, c = input.get_shape().as_list()
            up_input = tf.image.resize_nearest_neighbor(input, [self.scale * h, self.scale * w])
            output = tf.layers.conv2d(up_input, self.filters, kernel_size=self.k_size,
                                      activation=self.activation,
                                      padding='same',
                                      name=self.name,
                                      bias_regularizer=self.regularizer,
                                      kernel_regularizer=self.regularizer)
            return output

    print('CONV_LAYER_STRUCTURE', [i['filters'] if i['type'] == 'regu2d' else 0 for i in layers_meta])
    with tf.name_scope(name_scope) as scope:
        conv_layers = [input_tensor, ]
        for i in layers_meta:
            if i['type'] == 'regu2d':
                conv_layers.append(tf.layers.Conv2D(i['filters'], i['k_size'],
                                                    activation=i['func'],
                                                    kernel_regularizer=i['reg'],
                                                    bias_regularizer=i['reg'],
                                                    padding='same',
                                                    name=i['name']))
            elif i['type'] == 'pool':
                if 'name' not in i:
                    i['name'] = 'inner_layer'
                conv_layers.append(tf.layers.MaxPooling2D(i['k_size'],
                                                          i['strides'],
                                                          name=i['name']))
            elif i['type'] == 'trans2d':
                i = soft_add_attribute_to_dict(i, 'name', 'inner_layer')
                i = soft_add_attribute_to_dict(i, 'reg', None)
                i = soft_add_attribute_to_dict(i, 'strides', i['scale'])
                i = soft_add_attribute_to_dict(i, 'func', tf.nn.relu)
                conv_layers.append(tf.layers.Conv2DTranspose(i['filters'], i['k_size'],
                                                             strides=i['strides'],
                                                             activation=i['func'],
                                                             kernel_regularizer=i['reg'],
                                                             bias_regularizer=i['reg'],
                                                             padding='same',
                                                             name=i['name']))
            elif i['type'] == 'sub2d':
                i = soft_add_attribute_to_dict(i, 'name', 'inner_layer')
                i = soft_add_attribute_to_dict(i, 'reg', None)
                conv_layers.append(SubPixelUpScaling(i['scale'], i['k_size'], name=i['name'], regularizer=None))
            elif i['type'] == 'gan2d':
                i = soft_add_attribute_to_dict(i, 'name', 'inner_layer')
                i = soft_add_attribute_to_dict(i, 'reg', None)
                conv_layers.append(
                    ProgressiveGanUpsampling(i['scale'], i['filters'], i['k_size'], name=i['name'], regularizer=None))
            else:
                raise Exception("Wrong layer type in i_conv_layer")
        conv_layer_out = reduce(lambda lhs, rhs: rhs(lhs), conv_layers)

        # block_parameter_num = sum(map(lambda layer: layer.count_params(), conv_layers[1:]))
        # print('Number of parameters in ' + name_scope + ': ', block_parameter_num)
        return conv_layer_out


def random_split_data(data, label, proportion):
    """
    Split two numpy arrays into two parts of `proportion` and `1 - proportion`

    Args:
        - data: numpy array, to be split along the first axis
        - proportion: a float less than 1
    """
    assert data.shape[0] == label.shape[0]
    size = data.shape[0]
    s = np.random.permutation(size)
    split_idx = int(proportion * size)
    return data[s[:split_idx]], label[s[:split_idx]], data[s[split_idx:]], label[s[split_idx:]]
