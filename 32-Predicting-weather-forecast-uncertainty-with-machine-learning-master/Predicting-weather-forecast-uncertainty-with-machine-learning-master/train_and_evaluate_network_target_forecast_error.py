
# -*- coding: utf-8 -*-
"""

train and evaluate the network, trained on forecast error.

Here we use the best result from the gridsearch
Best:  {'train': 0.73619503620210269, 'valid': 0.27322227116573888}  using  {'conv_type': 'normal', 'drop_prob': 0.5, 'hidden_size': 512, 'lr': 0.0003}


"""

import matplotlib
matplotlib.use('agg')
import pickle
import os
import warnings
import numpy as np



## set random seeds for reproducability
import random as rn
import tensorflow as tf
import os

os.environ['PYTHONHASHSEED'] = '1'
np.random.seed(5)
rn.seed(1323)


from keras import backend as K
tf.set_random_seed(1241)


from sklearn import model_selection
from sklearn.model_selection import GridSearchCV, ParameterGrid
import sklearn
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten, concatenate, Activation
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.utils import plot_model
import keras.layers

from keras import optimizers
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.utils.training_utils import multi_gpu_model
import keras.callbacks


from sklearn.utils import shuffle

import pandas as pd
from scipy import signal

from keras.backend.tensorflow_backend import set_session
from pylab import plt
import seaborn as sns
import xarray as xr

from dask.diagnostics import ProgressBar
pbar = ProgressBar()
pbar.register()




#3 end set random seeds

# turn on tensorflow logging to check whether the GPU is used
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

N_GPU=2

def sellonlatbox(data,west,north,east,south):

    if north == south and west==east:
        return selclosestpoint(data,west,north)
    else:

        indexers = {'lon':slice(west,east), 'lat':slice(north,south)}

        sub = data.sel(**indexers)

        return sub


#%% PARAMS


startdate='198501010000'
enddate='201612310000'
subreg=[-165,0,60,80]  # 3 in teh gefs data, lat is from low to high
#subreg='NH'
#subreg=[-10,40,0,50]
subreg_string='_'.join([str(e) for e in subreg])


#indata_path ='/home/s/sebsc/pfs/data/GEFS_error_regression'
indata_path ='/proj/bolinc/users/x_sebsc/GEFS_error_regression/'
target_var='spread'

def read_data():
    ''' read in GEFS reforecast analysis and errors
    '''


    pklname=indata_path+'/'+str(subreg)+str(lead_day)+'.npy'
    pklname2=pklname+target_var+'.pkl'
    if os.path.exists(pklname) and os.path.exists(pklname2):
        print('restoring data form pkl')
        X = np.load(pklname)
        error_df = pickle.load(open(pklname2,'rb'))
    else:




        z_all = xr.open_mfdataset(indata_path+'/hgt_pres_latlon_c00_19841201_20171231_analysis_180-180.nc', chunks={'time':1})['Geopotential_height']

        u_all = xr.open_mfdataset(indata_path+'/ugrd_pres_latlon_c00_19841201_20171231_analysis_180-180.nc', chunks={'time':1})['U-component_of_wind']
        v_all = xr.open_mfdataset(indata_path+'/vgrd_pres_latlon_c00_19841201_20171231_analysis_180-180.nc', chunks={'time':1})['V-component_of_wind']

        # remove empty pressure dimension
        z_all = z_all.squeeze()
        u_all = u_all.squeeze()
        v_all = v_all.squeeze()

        assert(z_all.shape==u_all.shape)
        assert(z_all.shape==v_all.shape)
        #%% read in list list of pre-computed forecast errors and spread


        print(lead_day)
        error_df = pd.read_csv(indata_path+'/RMSE_gefsreforecasts_ctrl_and_spread_from_ordered_[-20, 80, 50, 20]_19850101_20161231_day'+str(lead_day)+'.csv', header=0,names=['date','ctrl','spread'], index_col=0)

        error_df.index = pd.DatetimeIndex(error_df.index)

        # subset time
        error_df = error_df[startdate:enddate]






        X_all = []
        for data, varname in ((z_all,'z500'),(u_all,'u300'),(v_all,'v300')):
            print('processing ',data )

            # cut to exactly same dates
            dates_y = error_df.index
            dates_X = data.time.to_pandas()
            common_dates = dates_y.intersection(dates_X)




            data = data.sel(time=common_dates)
            error_df = error_df[[e in common_dates for e in error_df.index]]

            if subreg is not 'NH':
                data = sellonlatbox(data,*subreg)


            assert(len(data.shape)==3)

            assert(len(data) == len(error_df))

            # load teh data
            data = np.array(data)

            X_all.append(data)

        X_all = np.stack(X_all,axis=3)
        X = X_all
        assert(len(X) == len(error_df))

        pickle.dump(error_df, open(pklname2,'wb'))
        np.save(pklname,X)

    return (X,error_df)



def correlation_coefficient_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))

    r = tf.truediv(r_num ,r_den)

    r = K.maximum(K.minimum(r, 1.0), -1.0)

    return 1 - K.abs(r)




lead_day = 3
print('start reading data')
X,y_all = read_data()
print('finished reading data')

# convert to 32 bit
X = X.astype('float32')
# detrend error (this also substracts the mean)

y_all['ctrl'] = signal.detrend(y_all['ctrl'])

y = y_all[target_var]


test_year=list(range(2010,2016+1))
valid_year=[1980,1990,2008]
print('splitting up data in test and training data')
print('selecting year ', test_year, ' for validation')
# X_train, X_test, y_train, y_test, dayofyear_train,dayofyear_test,dayofyear__norm_train,dayofyear_norm_test, errors_train, spreads_test = model_selection.train_test_split(
#         X, y, dayofyear,dayofyear_norm, spreads , test_size=test_size, random_state=random_state)


dates = y_all.index
# create boolean index
idcs_test = dates.year.isin(test_year)
idcs_valid = dates.year.isin(valid_year)

idcs_train = (~dates.year.isin(test_year)) & (~dates.year.isin(valid_year))
assert(np.array_equal((~idcs_train) & (~idcs_valid),idcs_test))
X_train, X_test, X_valid = X[idcs_train], X[idcs_test], X[idcs_valid]
y_train, y_test, y_valid = y[idcs_train], y[idcs_test], y[idcs_valid]


# normalize X Data per pixel, based on training data
for varidx in range(X.shape[-1]) :
    x = X_train[...,varidx]

    norm_mean = x.mean(axis=0)
    norm_std = x.std(axis=0)

    X_train[...,varidx] = (X_train[...,varidx] - norm_mean) / norm_std
    X_test[...,varidx] = (X_test[...,varidx] - norm_mean) / norm_std
    X_valid[...,varidx] = (X_valid[...,varidx] - norm_mean) / norm_std


# compute anomaly of y, based on training data
y_norm = y_train
y_tmp = y_norm.copy()
y_tmp.index.name = ['time']
y_xr = xr.DataArray(y_tmp)
clim_y = y_xr.rolling(time=30,center=True).mean().groupby('time.dayofyear').mean('time')

# expand to all days (replicating the days of the year dor each date in the data)
dummy = [clim_y.sel(dayofyear=m).values for m in xr.DataArray(y_train).date.dt.dayofyear]
y_train = y_train  - dummy

dummy = [clim_y.sel(dayofyear=m).values for m in xr.DataArray(y_test).date.dt.dayofyear]
y_test = y_test  - dummy

dummy = [clim_y.sel(dayofyear=m).values for m in xr.DataArray(y_valid).date.dt.dayofyear]
y_valid = y_valid  - dummy




# normalize y_data, based on training data
y_mean = y_train.mean()
y_std = y_train.std()

y_train =( y_train - y_mean ) / y_std
y_test =( y_test - y_mean ) / y_std
y_valid =( y_valid - y_mean ) / y_std




## build the network
batch_size = 100  ## note: if batch size is too small, I sometimes (or in factg quite often) got nans in the training process...
                    ## hoever, when useing whole NH data, the batch size cannot be 400 (the GPUs done have enough memory)
num_epochs = 30
kernel_size = 2
pool_size = 2
conv_depth_1 = 32
conv_depth_2 = 32




loss='mean_squared_error'



param_string='_'.join([str(e) for e in ('rmse_trained_on_new_anom',batch_size,num_epochs,kernel_size,pool_size,conv_depth_1,conv_depth_2,target_var,'subreg'+subreg_string,'lead_day'+str(lead_day))])


num_train, height, width, depth = X_train.shape
num_test = X_test.shape[0]




## define different convolution blocks (normal, batchnorm, residual)

def conv_normal(n_filter, w_filter ,inputs):
    return Activation(activation='relu')(Convolution2D(n_filter, w_filter, padding='same')(inputs))

def conv_batchnorm(n_filter, w_filter,  inputs):
    return BatchNormalization()(Activation(activation='relu')(Convolution2D(n_filter, w_filter,  padding='same')(inputs)))



conv_blocks = {'normal':conv_normal,'batchnorm':conv_batchnorm,
               #'residual':conv_residual,
               }

def create_model(lr,hidden_size,drop_prob,conv_type='normal', conv_depth1=32, conv_depth2=32):

    K.clear_session()  ## necessary because otherwise GPU memory will run full
    drop_prob_1 = drop_prob
    drop_prob_2 = drop_prob

    # use the specified convolution building block
    conv_block = conv_blocks[conv_type]


    inp = Input(shape=(height, width, depth)) # depth goes last in TensorFlow back-end (first in Theano)

    conv_1 = conv_block(conv_depth_1,kernel_size, inp)
    conv_2 = conv_block(conv_depth_1,kernel_size, conv_1)

    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
    drop_1 = Dropout(drop_prob_1)(pool_1)

    conv_3 = conv_block(conv_depth_2,kernel_size, drop_1)
    conv_4 = conv_block(conv_depth_2,kernel_size, conv_3)


    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
    drop_2 = Dropout(drop_prob_1)(pool_2)

    flat = Flatten()(drop_2)

    # # add second input for dayofyear
    # input2_size=1
    # input2 = Input(shape=[input2_size])
    #
    # merged = concatenate([flat,input2])


    hidden = Dense(hidden_size, activation='softmax')(flat)
    drop_3 = Dropout(drop_prob_2)(hidden)

    ## our output layer has just one neuron (because we do regression, not classification
    ## the last neuron should be linear for regression (https://www.reddit.com/r/MachineLearning/comments/4ebh0f/question_neural_networks_for_regression/)
    out = Dense(1, activation='linear')(drop_3)


    # we will use parallel GPUS for the training, therefore the setup has to be done using
    # the cpu
    with tf.device("/cpu:0"):
        model = Model(inputs=inp, outputs=out)
        # convert the model to a model that can be trained with N_GPU GPUs

        model = multi_gpu_model(model, gpus=N_GPU)

    print('compiling model')
    model.compile(loss=loss,
                  optimizer=optimizers.adam(lr=lr),
                  metrics=[correlation_coefficient_loss,'MSE'])

    return model



def score_func(y,y_pred):

    return np.abs(np.corrcoef(y,y_pred)[0,1])




scorer = sklearn.metrics.make_scorer(score_func, greater_is_better=True)



fit_params = {            'batch_size':batch_size,
                                        'epochs':num_epochs,'verbose':1,
                                        'validation_data':(X_valid,y_valid)
                                        }



# set params for the model (these are the parameters that came out of the gridsearch)
params= dict(lr=0.0003,
                    hidden_size=32,
                    drop_prob=0.7,
                    conv_type='normal',
                    conv_depth2=32)



output_overview_file = 'final_validation_result_overview'+param_string+'.txt'

results = []
fnames = []


print('start with params', params)



# we train the model N_tests time (this is necessary ebecause there is a lot
# of stochasticity in the training process)
N_tests = 10

for i in range(N_tests):

    # create model
    model = create_model(**params)

    # save initial weights
    model.save_weights('initial_weights_'+param_string+'_'+str(i)+'.h5')


    # we have to create the callbacks her in the loop, otherwise they are not reset
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=1), # if the model is not learning at all, we can stop
        keras.callbacks.TerminateOnNaN(),
        keras.callbacks.ModelCheckpoint('best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    ]
    # train model

    hist = model.fit(X_train,y_train, callbacks=callbacks,**fit_params)

    # svae the final model (this is usually not the best model, but it is useful if we want to train for more epochs)
    model_filename='final_model_weights_'+param_string+'_'+str(i)+'.h5'
    model.save_weights(model_filename)


    # get best model from the training (based on validation loss)
    model.load_weights('best_weights.h5')

    # remove the file created by ModelCheckppoint
    os.system('rm best_weights.h5')

    # save the best model
    model_filename='best_model_weights_'+param_string+'_'+str(i)+'.h5'
    model.save_weights(model_filename)

    y_valid_predicted = np.squeeze(model.predict(X_valid))
    y_train_predicted = np.squeeze(model.predict(X_train))
    y_test_predicted = np.squeeze(model.predict(X_test))

    valid_score = score_func(y_valid, y_valid_predicted)
    train_score = score_func(y_train, y_train_predicted)
    test_score = score_func(y_test, y_test_predicted)

    score = {'train': train_score, 'valid': valid_score, 'test': test_score}
    results.append((i, score))

    print('train:' + str(train_score) + ' valid:' + str(valid_score))

    # save model history and params
    hist_filename = 'history_'+param_string+'_'+str(i)+'.history.pkl'
    pickle.dump((hist.history,params), open(hist_filename,'wb'))

    # save predictions
    df_train = pd.DataFrame(dict(y_train_predicted=y_train_predicted,
                            y_train=y_train))
    df_test = pd.DataFrame(dict(y_test_predicted=y_test_predicted,
                            y_test=y_test))
    df_valid = pd.DataFrame(dict(y_valid_predicted=y_valid_predicted,
                            y_valid=y_valid))

    predictions_filename = 'predictions_'+param_string+'_' + str(i) + '.pkl'
    pickle.dump((df_train, df_test, df_valid), open(predictions_filename, 'wb'))

    # plot history

    df_hist = pd.DataFrame(hist.history)
    df_hist[['loss', 'val_loss']].plot()
    plt.figure()
    df_hist.plot()
    plt.savefig('hist_overview' + hist_filename + '.svg')

    # write to overview file
    with open(output_overview_file, 'a') as f:
        f.write('iteration ' + str(i) + ' valid:' + str(valid_score) + '\n')

    # scatterplot of results
    plt.figure()
    sns.jointplot(df_train['y_train_predicted'], df_train['y_train'], kind='reg', scatter_kws={'s':2,'alpha':0.6})
    plt.savefig('y_train_vs_y_train_predicted_'+str(i)+'_'+param_string+'.svg')

    plt.figure()
    sns.jointplot(df_test['y_test_predicted'], df_test['y_test'], kind='reg', scatter_kws={'s':2,'alpha':0.6})
    plt.savefig('y_test_vs_y_test_predicted_'+str(i)+'_'+param_string+'.svg')

    plt.figure()
    sns.jointplot(df_valid['y_valid_predicted'], df_valid['y_valid'], kind='reg', scatter_kws={'s':2,'alpha':0.6})
    plt.savefig('y_valid_vs_y_valid_predicted_'+str(i)+'_'+param_string+'.svg')

    plt.close('all')
