# Generate product-related features

import pandas as pd
import numpy as np


print('computing usersXproduct f')

orders_trim = pd.read_pickle('../processed_data/orders_trim.pickle')
priors_trim = pd.read_pickle('../processed_data/priors_trim.pickle')

# unique userXproduct id
priors_trim['user_product'] = priors_trim.product_id + priors_trim.user_id * 100000


userXproduct = pd.DataFrame()
userXproduct['up_first_two_reordered'] = priors_trim.loc[lambda priors_trim: priors_trim.is_first_two , :]\
                                                    .groupby('user_product')['reordered'].sum().astype(np.int16)
userXproduct['up_last_two_reordered'] = priors_trim.loc[lambda priors_trim: priors_trim.is_last_two , :]\
                                                   .groupby('user_product')['reordered'].sum().astype(np.int16)


# get userXproduct last order id
last_order_idx = priors_trim.groupby(['user_product'])['order_number'].transform(max) == priors_trim['order_number']
up_last_order = priors_trim[last_order_idx][['user_product', 'order_id']].set_index('user_product')
up_last_order.columns =['last_order_id']
up_last_order = (up_last_order.reset_index()).drop_duplicates(subset='user_product').set_index('user_product')



user_product = priors_trim.groupby('user_product').agg({'user_product': 'count', 'order_number': [min, max], 'order_dow': 'mean',                                             'order_hour_of_day': 'mean', 'add_to_cart_order': 'mean',                                              'days_since_prior_order': ['mean', 'std'], 'days_is_30': 'sum' })
user_product.columns = ["_".join(x) for x in user_product.columns.ravel()]
user_product = user_product.join(userXproduct, how='left')

userXproduct = user_product.join(up_last_order, how='left')
del user_product
del up_last_order

userXproduct['order_rate_since_first'] = userXproduct['user_product_count'] / (userXproduct['order_number_max'] - userXproduct['order_number_min'] + 1)
userXproduct['up_weekend_number'] = priors_trim.groupby('user_product')['weekend'].sum().astype(np.int16)
userXproduct['up_weekly_number'] = priors_trim.groupby('user_product')['weekly'].sum().astype(np.int16)
userXproduct['up_weekend_ratio'] = userXproduct['up_weekend_number'] / userXproduct['user_product_count']
userXproduct['up_weekly_ratio'] = userXproduct['up_weekly_number'] / userXproduct['user_product_count']

# save pickles
priors_trim.to_pickle('../processed_data/priors_trim.pickle')
orders_trim.to_pickle('../processed_data/orders_trim.pickle')
userXproduct.to_pickle('../processed_data/feature_userXproduct.pickle')


