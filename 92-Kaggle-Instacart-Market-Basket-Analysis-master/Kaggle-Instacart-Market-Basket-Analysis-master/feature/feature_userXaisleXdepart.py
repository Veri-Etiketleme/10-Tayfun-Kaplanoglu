# Generate product-related features

import pandas as pd
import numpy as np


print('computing userXaisle and userXdepart f')

priors_trim = pd.read_pickle('../processed_data/priors_trim.pickle')

print('compute userXaisle f ')

priors_trim['user_aisle'] = priors_trim.aisle_id + priors_trim.user_id * 100000

userXaisle = pd.DataFrame()
userXaisle['ua_orders'] = priors_trim.groupby('user_aisle').size()
userXaisle['ua_reorder'] = priors_trim.groupby('user_aisle')['reordered'].sum()
userXaisle['ua_reorder_rate'] = userXaisle.ua_reorder /  userXaisle.ua_orders
userXaisle['ua_avg_days_since'] = priors_trim.groupby('user_aisle')['days_since_prior_order'].mean()

print('compute userXdepart f ')

priors_trim['user_depart'] = priors_trim.department_id + priors_trim.user_id * 100000

userXdepart = pd.DataFrame()
userXdepart['ud_orders'] = priors_trim.groupby('user_depart').size()
userXdepart['ud_reorder'] = priors_trim.groupby('user_depart')['reordered'].sum()
userXdepart['ud_reorder_rate'] = userXdepart.ud_reorder /  userXdepart.ud_orders
userXdepart['ud_avg_days_since'] = priors_trim.groupby('user_depart')['days_since_prior_order'].mean()

# save pickles
userXaisle.to_pickle('../processed_data/feature_userXaisle.pickle')
userXdepart.to_pickle('../processed_data/feature_userXdepart.pickle')


