"""

Due to the file size, I sampled 20% of the the users and their orders history.

"""

import pandas as pd
import numpy as np


print('Get trimmed dataframes with 20% of training data')

orders = pd.read_csv('../input/orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})

priors = pd.read_csv('../input/order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})

products = pd.read_csv('../processed_data/products_PCA.csv', encoding = "ISO-8859-1")


# use 20% of the training data as training set
orders_train = orders[orders['eval_set'] == 'train']
orders_train_trim = orders_train.sample(frac=0.2, random_state=42)


# get a trimmed orders df
orders_test = orders[orders['eval_set'] == 'test']
train_user_id = orders_train_trim['user_id'].tolist()
test_user_id = orders_test['user_id'].tolist()
trim_user_id = train_user_id + test_user_id
orders_trim = orders.loc[orders['user_id'].isin(trim_user_id)]


# get a trimmed priors order df and merge the products information to the priors df
trim_order_id = orders_trim['order_id'].tolist()
priors_trim = priors.loc[priors['order_id'].isin(trim_order_id)]
priors_trim = priors_trim.merge(products, on='product_id', how='left')

# save to pickle
priors_trim.to_pickle('../processed_data/priors_trim.pickle')
orders_trim.to_pickle('../processed_data/orders_trim.pickle')