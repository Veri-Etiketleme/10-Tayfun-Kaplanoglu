import numpy as np 
import tensorflow as tf
from CNN_class import * 
import argparse
import math

import pdb

def dataparse(filename, window):
    data = np.loadtxt(filename, delimiter=',')
    m, n = data.shape
    X = np.zeros([m-window,window, n-1,1])
    Y = np.zeros([m-window,1]) 
    for i in range(m-window):
        Y[i,:] = (data[i,0] + 1)/2
        j0 = i
        j1 = i + window
        X[i,:,:,0] = data[j0:j1, 1:]
    return X, Y

def main(args):
    Xd, Yd = dataparse('./Data/test_data_class.csv', args.window)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, args.window, 8, 1])
    Y = tf.placeholder(tf.float32, [None, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('ResNet') as enc: 
        Y_ = resnet(X, is_training)
    
    sess = tf.Session()
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    model_name = './class_params/CNN_class'
    saver.restore(sess, model_name)

    P = 30
    feed_dict = {X:Xd[0:P,...], Y:Yd[0:P,:], is_training:False}
    Ye = sess.run(Y_, feed_dict=feed_dict)
    selection = np.argmax(Ye, axis=1)
    correct = selection == Yd[0:P,:].squeeze()
    accuracy = np.mean(correct)
    print('Accuracy: %r' % accuracy)
    import pdb; pdb.set_trace()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('window', type=int)
    args = parser.parse_args()
    main(args)
