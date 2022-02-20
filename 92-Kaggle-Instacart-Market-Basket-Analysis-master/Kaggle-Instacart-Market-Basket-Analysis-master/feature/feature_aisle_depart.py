# Generate product-related features

import pandas as pd
import numpy as np


priors_trim = pd.read_pickle('../processed_data/priors_trim.pickle')
aisles_df = pd.read_csv('../input/aisles.csv')
departments_df = pd.read_csv('../input/departments.csv')


print('computing aisle f')
aisle = pd.DataFrame()
aisle['as_orders'] = priors_trim.groupby('aisle_id').size().astype(np.int32)
aisle['as_reorders'] = priors_trim.groupby('aisle_id')['reordered'].sum().astype(np.int32)
aisle['as_reorder_rate'] = aisle.as_reorders / aisle.as_orders.astype(np.float32)
aisle['as_avg_days_since'] = priors_trim.groupby('aisle_id')['days_since_prior_order'].mean()
aisle['as_unique_user'] = priors_trim.groupby('aisle_id')['user_id'].nunique()
aisle['as_order_per_unique_user'] = aisle.as_orders / aisle.as_unique_user
aisle['as_avg_add_to_cart'] = priors_trim.groupby('aisle_id')['add_to_cart_order'].mean()
aisle['as_unique_prod'] = priors_trim.groupby('aisle_id')['product_id'].nunique()


print('computing department f')
depart = pd.DataFrame()
depart['dp_orders'] = priors_trim.groupby('department_id').size().astype(np.int32)
depart['dp_reorders'] = priors_trim.groupby('department_id')['reordered'].sum().astype(np.int32)
depart['dp_reorder_rate'] = depart.dp_reorders / depart.dp_orders.astype(np.float32)
depart['dp_avg_days_since'] = priors_trim.groupby('department_id')['days_since_prior_order'].mean()
depart['dp_unique_user'] = priors_trim.groupby('department_id')['user_id'].nunique()
depart['dp_order_per_unique_user'] = depart.dp_orders / depart.dp_unique_user
depart['dp_avg_add_to_cart'] = priors_trim.groupby('department_id')['add_to_cart_order'].mean()
depart['dp_unique_prod'] = priors_trim.groupby('department_id')['product_id'].nunique()


# save pickles
aisle.to_pickle('../processed_data/feature_aisle.pickle')
depart.to_pickle('../processed_data/feature_depart.pickle')


