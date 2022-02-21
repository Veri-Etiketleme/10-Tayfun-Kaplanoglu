# Generate product-related features

import pandas as pd
import numpy as np


print('computing users f')

orders_trim = pd.read_pickle('../processed_data/orders_trim.pickle')
priors_trim = pd.read_pickle('../processed_data/priors_trim.pickle')
user_rfidf = pd.read_csv('../processed_data/user_rfidf.csv')

# total orders for each user id
orders_trim['max_order'] = orders_trim['user_id'].map(orders_trim.groupby('user_id').order_number.max())

# get user features from orders_trim
users = pd.DataFrame()
users['first_avg_days_between_orders'] = orders_trim.loc[lambda orders_trim: orders_trim.order_number.isin([1,2,3]) , :]                                                   .groupby('user_id')['days_since_prior_order'].mean()
users['last_avg_days_between_orders'] = orders_trim.loc[lambda orders_trim: orders_trim.order_number > orders_trim.max_order - 3 , :]                                                   .groupby('user_id')['days_since_prior_order'].mean()
users['last_first_avg_days_ratio'] = users['last_avg_days_between_orders'] / users['first_avg_days_between_orders']
users['diff_hour_since_last'] = orders_trim.loc[lambda orders_trim: orders_trim.order_number > orders_trim.max_order - 2 , :]                                                   .groupby('user_id')['order_hour_of_day'].agg(np.ptp)
users['average_days_between_orders'] = orders_trim.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
users['std_days_between_orders'] = orders_trim.groupby('user_id')['days_since_prior_order'].std().astype(np.float32)
users['nb_orders'] = orders_trim.groupby('user_id').size().astype(np.int16)
users['avg_order_dow'] = orders_trim.groupby('user_id')['order_dow'].mean().astype(np.float32)
users['mode_order_dow'] = orders_trim.groupby('user_id')['order_dow'].agg(lambda x:x.value_counts().index[0]).astype(np.float32)
users['mode_order_hour_of_day'] = orders_trim.groupby('user_id')['order_hour_of_day'].agg(lambda x:x.value_counts().index[0]).astype(np.float32)
users['avg_order_hour_of_day'] = orders_trim.groupby('user_id')['order_hour_of_day'].mean().astype(np.float32)
users['tenure'] = orders_trim.groupby('user_id')['days_since_prior_order'].sum()
users['avg_nb_tenure'] = users['nb_orders'] / users['tenure']
users['nb_weekend_order'] = orders_trim.groupby('user_id')['weekend'].sum().astype(np.int16)
users['nb_weekly_order'] = orders_trim.groupby('user_id')['weekly'].sum().astype(np.int16)
users['weekend_ratio'] = (users['nb_weekend_order'] / users['nb_orders']).astype(np.float32)
users['weekly_ratio'] = (users['nb_weekly_order'] / users['nb_orders']).astype(np.float32)
users['user_nb_days_is_30'] = orders_trim.groupby('user_id')['days_is_30'].sum().astype(np.int16)
users['user_nb_days_is_30_ratio'] = (users['user_nb_days_is_30'] / users['nb_orders']).astype(np.float32)

# check and flag the first and last orders for each user
priors_trim['max_order_prior'] = priors_trim['user_id'].map(priors_trim.groupby('user_id').order_number.max())
priors_trim['is_last_two'] = False
priors_trim['is_first_two'] = False
priors_trim.loc[lambda priors_trim: priors_trim.order_number > priors_trim.max_order_prior - 1 , 'is_last_two'] = True
priors_trim.loc[lambda priors_trim: priors_trim.order_number.isin([2,3]) , 'is_first_two'] = True

users['first_two_items_number'] = priors_trim.loc[lambda priors_trim: priors_trim.is_first_two , :].groupby('user_id').size().astype(np.int16)
users['last_two_items_number'] = priors_trim.loc[lambda priors_trim: priors_trim.is_last_two , :].groupby('user_id').size().astype(np.int16)
users['first_two_reorder_number'] = priors_trim.loc[lambda priors_trim: priors_trim.is_first_two , :].groupby('user_id')['reordered'].sum().astype(np.int16)
users['last_two_reorder_number'] = priors_trim.loc[lambda priors_trim: priors_trim.is_last_two , :].groupby('user_id')['reordered'].sum().astype(np.int16)
users['first_last_number_ratio'] = users['last_two_items_number'] / users['first_two_items_number']
users['first_last_reorder_ratio'] = users['last_two_reorder_number'] / users['first_two_reorder_number']


# get user features from prior_trim
users['total_items'] = priors_trim.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors_trim.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
users['distinct_ratio'] = (users['total_distinct_items'] / users['total_items']).astype(np.float32)
users['reordered_number'] = priors_trim.groupby('user_id')['reordered'].sum()
users['last2_all_reorder_ratio'] = users['last_two_reorder_number'] / users['reordered_number']
users['last_product_count'] = priors_trim.loc[lambda priors_trim: priors_trim.order_number == priors_trim.max_order_prior , :]                                            .groupby('user_id').size().astype(np.int16)
users['last_reorder_count'] = priors_trim.loc[lambda priors_trim: priors_trim.order_number == priors_trim.max_order_prior , :]                                            .groupby('user_id')['reordered'].sum().astype(np.int16)
users['last_reorder_rate'] = users['last_reorder_count'] / users['last_product_count']
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
users['last_vs_previous_basket'] = (users.last_product_count - users.average_basket) / users.average_basket

# combine the features from user_rfidf
user_rfidf.set_index('user_id', inplace=True)
users = users.join(user_rfidf, how='left')

# save pickles
priors_trim.to_pickle('../processed_data/priors_trim.pickle')
orders_trim.to_pickle('../processed_data/orders_trim.pickle')
users.to_pickle('../processed_data/feature_users.pickle')


