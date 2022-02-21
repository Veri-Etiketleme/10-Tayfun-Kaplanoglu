import numpy as np 
import tensorflow as tf
from CNN import * 
import argparse
import math

import pdb

def dataparse(filename, y_idx, window):
    data = np.loadtxt(filename, delimiter=',')
    m, n = data.shape
    X = np.zeros([m-window,window, n-5,1])
    Y = np.zeros([m-window,1]) 
    for i in range(m-window):
        Y[i,:] = data[i,y_idx]
        j0 = i
        j1 = i + window
        X[i,:,:,0] = data[j0:j1, 5:]
    return X, Y 

def main(args):
    Xd, Yd = dataparse('./Data/train_data.csv', args.y, args.window)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, args.window, 7, 1])
    Y = tf.placeholder(tf.float32, [None, 1])
    is_training = tf.placeholder(tf.bool)
    
    with tf.variable_scope('ResNet') as enc: 
        Y_ = resnet(X, is_training)

    # import pdb; pdb.set_trace()
    l2_loss = (Y - Y_)**2
    # l1_loss = tf.abs(Y - Y_)
    sum_loss = l2_loss #+ l1_loss
    mean_loss = tf.reduce_mean(sum_loss)
    tf.summary.scalar('l2_loss', tf.reduce_mean(l2_loss))
    # tf.summary.scalar('l1_loss', tf.reduce_mean(l1_loss))
    tf.summary.scalar('loss', mean_loss)

    optimizer = tf.train.AdamOptimizer(learning_rate=args.rate)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train = optimizer.minimize(mean_loss)
    
    sess = tf.Session()
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./tb',sess.graph)

    sess.run(tf.global_variables_initializer())

    _ = run_model(sess, X, Y, is_training, mean_loss, Xd, Yd, 
              epochs=args.epochs, print_every=100, 
              training=train, plot_losses=False,
              writer=writer, sum_vars=merged)

    feed_dict = {X:Xd, Y:Yd, is_training:False}
    Ye = sess.run(Y_, feed_dict=feed_dict)
    delta = np.mean((Ye - Yd)**2/Yd**2)
    # class_accuracy = np.mean(np.sign(Yd[:,0]) == np.sign(Ye[:,0]))
    print('Average Error: %r' % delta)
    import pdb; pdb.set_trace()
    model_name = './params/CNN'
    saver.save(sess, model_name)
    print('Training complete. Model saved. Exiting')

def run_model(session, X, Y, is_training, loss_val, Xd, Yd, 
              epochs=1, batch_size=64, print_every=100,
              training=None, plot_losses=False,writer=None, sum_vars=None):
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
    # setting up variables we want to compute (and optimizing)
    # if we have a training function, add that to things we compute
    variables = [loss_val, training]
    if writer is not None: 
        variables.append(sum_vars)

    # counter 
    iter_cnt = 0
    for e in range(epochs):
        losses = []
        
        # make sure we iterate over the dataset once
        for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%Xd.shape[0]
            idx = train_indicies[start_idx:start_idx+batch_size]
            
            # create a feed dictionary for this batch
            feed_dict = {X: Xd[idx,:],
                         Y: Yd[idx,:],
                         is_training: True}
            # get batch size
            actual_batch_size = Yd[i:i+batch_size].shape[0]
            
            # have tensorflow compute loss and correct predictions
            # and (if given) perform a training step
            if writer is not None:
                # import pdb; pdb.set_trace()
                loss, _, summary = session.run(variables,feed_dict=feed_dict)
                writer.add_summary(summary, iter_cnt)
            else:
                loss, _ = session.run(variables,feed_dict=feed_dict)
            # aggregate performance stats
            losses.append(loss*actual_batch_size)
            
            # print every now and then
            if (iter_cnt % print_every) == 0:
                print("Iteration %r: with minibatch training loss = %r " % (iter_cnt,loss))
            iter_cnt += 1
        total_loss = np.sum(losses)/Xd.shape[0]
        print("Epoch {1}, Overall loss = {0:.3g}"\
              .format(total_loss,e+1))
        if plot_losses:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.show()
    return total_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test CNN translation for given arguments')
    parser.add_argument('y', type=int)
    parser.add_argument('window', type=int)
    parser.add_argument('epochs', type=int)
    parser.add_argument('batch_size', type=int) 
    parser.add_argument('rate', type=float) 
    args = parser.parse_args()
    main(args)
