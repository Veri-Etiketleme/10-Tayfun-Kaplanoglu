"""
Define functions for manipulated the feature data

"""
import sys
from datetime import date, timedelta
import string as s
from copy import deepcopy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm, preprocessing
from sklearn.neural_network import MLPClassifier

import pdb


YAHOO_FILES = ['DJIA.csv', 'SP500.csv', 'Nikkei225.csv', 'Stoxx600.csv',
               'VIX.csv', 'CEW.csv', 'BZF.csv']
DESIRED_COLS = ['Date', 'Close']
AU_DATA = '../Network/PCAData/AU_data.csv'
AU_START = 1283767114
LT_DATA = '../Network/PCAData/lt_data.csv'
LT_START = 1276427213
ONE_COLUMN_DATA = ['price.csv']
DATA_TO_INTERPOLATE=['GoogleTrends_Bitcoin.csv', 'GoogleTrends_BitcoinNews.csv',
                     'GoogleTrends_BitcoinPrice.csv']
NUM_OUTPUT = 3


def read_repeated_columns(fn_list, col_list, interpolate=False):
    """ Merges data found in multiple csv files (e.g. closing and high values
        for the DJIA and SP500) into a single DataFrame, keeping only dates
        found in all datasets.
        """
    table_list = []
    for (i, fn) in enumerate(fn_list):

        # Get the data for the current file into a DataFrame
        table_list.append(pd.read_csv(fn, parse_dates=[0], index_col=0,
                                      usecols=col_list,
                                      infer_datetime_format=True))
        if interpolate is True:
            data = table_list[i]
            date_index = pd.date_range(data.index[0], data.index[-1])
            data = data.reindex(date_index)
            table_list[i] = data.interpolate(method='time')

        # merge this new DataFrame into the combined DataFrame
        if i is 0:
            table_cat = table_list[i]
        else:
            table_cat = pd.merge(table_cat, table_list[i], left_index=True,
                                 right_index=True, how='inner',
                                 suffixes=[fn_list[i-1].rstrip('.csv'),
                                           fn_list[i].rstrip('.csv')])

    # Relabel the last columns, which will not have been automatically
    # label mangled when fn_list has an odd number of files
    if col_list is not None:
        new_labels = table_cat.columns.values
        for (i, _) in enumerate(col_list[1:]):
            new_labels[-1-i] += fn_list[-1].rstrip('.csv')
        table_cat.columns = new_labels
    return table_cat


def read_graph_data(fn, name, cols, start_stamp):
    """ Read in the BTC graph data in Mern's format into the format for the
        rest of the data (date column, then data columns) """
    # Read in the data
    data = pd.read_csv(fn, usecols=([0] + cols) )

    # Convert integer indices into datetime objects
    start = date(1,1,1).fromtimestamp(start_stamp)
    dt = timedelta(days=1)
    data.iloc[:,0] = data.iloc[:,0]*dt + start

    # Keep only the first singular vector, and give the data column labels
    names = [name + str(i) for i in cols]
    data.columns = ['Date'] + names
    data = data.set_index('Date')
    return data


def add_shifted_cols(data, shift_num, suffix=None, include_orig=False):
    """ Adds shifted versions of all the columns in data """
    shifted = data.shift(shift_num)
    new_labels = deepcopy(shifted.columns.values)
    for (i, _) in enumerate(new_labels):
        new_labels[i] += suffix
    shifted.columns = new_labels
    if include_orig:
        shifted = pd.concat([data, shifted], join="inner", axis=1)
    return shifted


def combine_all_data():
    """ Combines all of our data into one feature matrix, doing the necessary
        data handling for each set of features """
    # Put all the features into the table
    t = read_repeated_columns(YAHOO_FILES, DESIRED_COLS)
    au = read_graph_data(AU_DATA, 'active_users', [1,2,3,4], AU_START)
    lt = read_graph_data(LT_DATA, 'lt_users', [1,2,3,4], LT_START)
    trends = read_repeated_columns(DATA_TO_INTERPOLATE, None, interpolate=True)
    btc = read_repeated_columns(ONE_COLUMN_DATA, None)
    all_features = pd.concat([t, lt, au, trends, btc], join="inner", axis=1)

    # Add yesterday's value of each feature to the column
    with_yesterday = add_shifted_cols(all_features, 1, "_prev", True)

    # Add columns for EMAs for BTC price
    with_yesterday[["signal_line"]] = with_yesterday[["Bitcoin Price"]].ewm(span=9).mean()
    with_yesterday[["MACD"]] = (with_yesterday[["Bitcoin Price"]].ewm(span=12).mean() -
                                with_yesterday[["Bitcoin Price"]].ewm(span=26).mean() )

    # Add column for tomorrow's BTC price
    tmrw = add_shifted_cols(with_yesterday[["Bitcoin Price"]], -1, "_tmrw", False)
    with_tmrw = pd.concat([with_yesterday, tmrw], join="inner", axis=1)

    # Add column for fractional change in BTC price
    with_tmrw["frac_change"] = ((with_tmrw["Bitcoin Price_tmrw"] -
                                 with_tmrw["Bitcoin Price"]) /
                                 with_tmrw["Bitcoin Price"])
    with_tmrw[["up_down"]] = ((with_tmrw[["frac_change"]]+0.001)/
                              (2*abs(with_tmrw[["frac_change"]]+0.001)) + 0.5)
    return with_tmrw.dropna(axis=0, how='any')


def lasso():
    """ Does the lasso method to downselect samples """
    return


def do_PCA(X, num_out=None, vec_file=None, trans_file=None ):
    """ Does PCA  """
    pca = PCA(num_out)
    pca.fit(X)
    np.savetxt(vec_file, pca.components_, delimiter=',')
    np.savetxt(trans_file, pca.fit_transform(X), delimiter=',')
    return pca


def score_classifier(clf, X, frac_change):
    """ Computes the ROI for a buy-don't classifier given a feature matrix,
        classifier, and true output vector """
    pred = clf.predict(X)
    d = dict()
    d["No Action"] = np.prod(1 + frac_change)
    d["Perfect Trading"] = np.prod(1 + frac_change[frac_change > 0])
    buy_days = np.multiply(frac_change, pred)
    d["This Classifier"] = np.cumprod(1 + buy_days)
    pdb.set_trace()
    return d


# Run from the command line, with the usage
#    python process.py [write_name]
# where [write_name] is an optional argument indicating the name of
# a file to write the merged dataframe into (as a .csv)
if __name__ == "__main__":

    comb = combine_all_data()
    means = comb.rolling(window=5, center=False).mean()
    comb = comb.drop(labels=[
                               'lt_users1', 'lt_users2', 'lt_users3', 'lt_users4',
                              'lt_users1_prev', 'lt_users2_prev', 'lt_users3_prev', 'lt_users4_prev',
                              'active_users1', 'active_users2', 'active_users3', 'active_users4',
                              'active_users1_prev', 'active_users2_prev', 'active_users3_prev', 'active_users4_prev',
                              'bitcoin_trends', 'bitcoinNews_trends', 'bitcoinPrice_trends',
                              'bitcoin_trends_prev', 'bitcoinNews_trends_prev', 'bitcoinPrice_trends_prev'
                              ],
                      axis=1)

    # Do PCA
    (num_ex, num_feat) = comb.values.shape
    num_feat -= NUM_OUTPUT     # last three columns are outputs
    X = comb.values[:, :num_feat-1]
    do_PCA(X, vec_file='singular_vectors.csv',
           trans_file='tranformed_data.csv')

    # Partition the data
    X = preprocessing.scale(X)
    frac_change = np.squeeze(comb[["frac_change"]].values)
    up_down = np.squeeze(comb[["up_down"]].values)
    price_tmrw = np.squeeze(comb[["Bitcoin Price_tmrw"]].values)
    #X_train, X_test, y_train, y_test = train_test_split(
    #     X, np.concatenate((np.expand_dims(up_down, axis=1),
    #                        np.expand_dims(frac_change, axis=1)), axis=1),
    #     test_size=0.3, random_state=0)
    trainind = np.genfromtxt('trainInd.csv', delimiter=',', dtype=int) - 1
    allind = list(range(X.shape[0]))
    testind = list(set(allind) - set(trainind))
    X_train = X[trainind, :]
    X_test = X[testind, :]
    ud_train =up_down[trainind]
    frac_change_train = frac_change[trainind]
    ud_test = up_down[testind]
    frac_change_test = frac_change[testind]

    # Try SVM
    param_grid = [
      {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 200, 1000, 2e3, 3e3, 4e3], 'gamma': [1e-3, 1e-4, 1e-5, 5e-6, 1e-7], 'kernel': ['rbf']},
      ]
    clf = GridSearchCV(svm.SVC(), param_grid)
    #clf = MLPClassifier(hidden_layer_sizes=(20,20),
    #                    activation='logistic',
    #                    solver='adam',
    #                    random_state=10,
    #                    alpha = 1e-2 )
    clf.fit(X_train, ud_train)
    clf.score(X_test, ud_test)
    score_dict = score_classifier(clf, X_test, frac_change_test)

    pdb.set_trace()

    #means.plot(subplots=True)
    #plt.show()
    print(comb)
    if len(sys.argv) > 1:
        comb.to_csv(sys.argv[1])
        if (len(sys.argv) > 2):
            means.to_csv(sys.argv[2])

    # now run linear regression