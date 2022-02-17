import numpy as np 
import tensorflow as tf 

def bn_conv2d(X, is_training, filters, kernel_size, strides=1, padding='valid',activation='relu', name=None):
    if name is not None: 
        with tf.variable_scope(name):
            c1 = tf.layers.conv2d(
                                inputs=X, 
                                filters=filters,
                                kernel_size=kernel_size,
                                strides=strides,
                                padding=padding,
                                activation=None,
                                name='c1')
            bn1 = tf.layers.batch_normalization(
                            c1, training=is_training,
                            renorm=True,  
                            name='bn1')
            if activation == 'relu':
                h1 = tf.nn.relu(bn1, name='h1')
            elif activation =='tanh':
                h1 = tf.tanh(bn1, name='h1')
            else: 
                h1 = bn1
    else:
        c1 = tf.layers.conv2d(
                            inputs=X, 
                            filters=filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding=padding,
                            activation=None,
                            name='c1')
        bn1 = tf.layers.batch_normalization(
                            c1, training=is_training,
                            renorm=True,  
                            name='bn1')
        if activation == 'relu':
            h1 = tf.nn.relu(bn1, name='h1')
        elif activation =='tanh':
            h1 = tf.tanh(bn1, name='h1')
        else: 
            h1 = bn1
    return h1


def resnet(X, is_training):
    p = X._shape_as_list()[2]
    c0 = bn_conv2d(X, is_training, filters=64, kernel_size=[3,p], strides=1, padding='valid',activation='relu', name='strided0')
    c1 = bn_conv2d(c0, is_training, filters=128, kernel_size=[3,1], strides=1, padding='same',activation='relu', name='strided1')
    c2 = bn_conv2d(c1, is_training, filters=256, kernel_size=[3,1], strides=1, padding='same',activation='relu', name='strided2')
    c3 = bn_conv2d(c2, is_training, filters=512, kernel_size=[8,1], strides=1, padding='valid',activation='relu', name='strided3')
    h0 = tf.squeeze(c3,axis=[1,2])
    h1 = tf.layers.dense(h0, 256, activation=tf.nn.relu, use_bias=False, name='FC0')
    h2 = tf.layers.dense(h1, 128, activation=tf.nn.relu, use_bias=False, name='FC1')
    h3 = tf.layers.dense(h2, 64, activation=tf.nn.relu, use_bias=False, name='FC2')
    h4 = tf.layers.dense(h3, 32, activation=tf.nn.relu, use_bias=False, name='FC3')
    y = tf.layers.dense(h4, 1, activation=None, use_bias=False, name='Out')
    return y
