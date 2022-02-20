# Generate product-related features

import pandas as pd
import numpy as np


print('computing product f')

orders_trim = pd.read_pickle('../processed_data/orders_trim.pickle')
priors_trim = pd.read_pickle('../processed_data/priors_trim.pickle')
products = pd.read_csv('../processed_data/products_PCA.csv', encoding = "ISO-8859-1")
aisles_df = pd.read_csv('../input/aisles.csv')
departments_df = pd.read_csv('../input/departments.csv')


#  Temporal features for the orders
orders_trim['days_is_30'] = orders_trim['days_since_prior_order'].apply(lambda x: 1 if x == 30 else 0)
orders_trim["weekend"] = orders_trim["order_dow"].apply(lambda x:1 if x in [0,1] else 0)
orders_trim["weekly"] = orders_trim["days_since_prior_order"].apply(lambda x:1 if x in [7,14,21,28] else 0)

orders_trim.set_index('order_id', inplace=True, drop=False)
priors_trim = priors_trim.join(orders_trim, on='order_id', rsuffix='_')
priors_trim.drop('order_id_', inplace=True, axis=1)

#priors_trim["weekend"] = priors_trim["order_dow"].apply(lambda x:1 if x in [0,1] else 0)
#priors_trim["weekly"] = priors_trim["days_since_prior_order"].apply(lambda x:1 if x in [7,14,21,28] else 0)

# products features
prods = pd.DataFrame()
prods['orders'] = priors_trim.groupby(priors_trim.product_id).size().astype(np.int32)
prods['reorders'] = priors_trim['reordered'].groupby(priors_trim.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
prods = prods.reset_index()
products = products.merge(prods, on='product_id', how='left')
products.set_index('product_id', drop=False, inplace=True)
del prods

# merge with depart & aisle
products = products.merge(departments_df, on='department_id', how='left')
products = products.merge(aisles_df, on='aisle_id', how='left')
products.set_index('product_id', inplace=True, drop=False)


# temporal features for products
products['prod_nb_weekend_order'] = priors_trim.groupby('product_id')['weekend'].sum().astype(np.int16)
products['prod_nb_weekly_order'] = priors_trim.groupby('product_id')['weekly'].sum().astype(np.int16)
products['prod_nb_days_is_30'] = priors_trim.groupby('product_id')['days_is_30'].sum().astype(np.int16)
products['prod_weekend_ratio'] = (products['prod_nb_weekend_order'] / products['orders']).astype(np.float32)
products['prod_weekly_ratio'] = (products['prod_nb_weekly_order'] / products['orders']).astype(np.float32)
products['prod_days_30_ratio'] = (products['prod_nb_days_is_30'] / products['orders']).astype(np.float32)

products['order_in_cart'] = priors_trim.groupby('product_id')['add_to_cart_order'].mean().astype(np.int16)
products['prod_avg_dow'] = priors_trim.groupby('product_id')['order_dow'].mean().astype(np.float32)
products['prod_avg_hour_of_day'] = priors_trim.groupby('product_id')['order_hour_of_day'].mean().astype(np.float32)
products['prod_avg_days_since'] = priors_trim.groupby('product_id')['days_since_prior_order'].mean().astype(np.float32)
products['prod_avg_order_number'] = priors_trim.groupby('product_id')['order_number'].mean().astype(np.float32)
products['prod_std_days_since'] = priors_trim.groupby('product_id')['days_since_prior_order'].std().astype(np.float32)
products['prod_nb_weekend_order'] = priors_trim.groupby('product_id')['weekend'].sum().astype(np.int16)
products['prod_nb_weekly_order'] = priors_trim.groupby('product_id')['weekly'].sum().astype(np.int16)
products['prod_weekend_ratio'] = (products['prod_nb_weekend_order'] / products['orders']).astype(np.float32)
products['prod_weekly_ratio'] = (products['prod_nb_weekly_order'] / products['orders']).astype(np.float32)

products = products.rename(columns={'Dimension 1': 'dimension_1', 'Dimension 2': 'dimension_2', 'Dimension 3': 'dimension_3'})

# save pickles
priors_trim.to_pickle('../processed_data/priors_trim.pickle')
orders_trim.to_pickle('../processed_data/orders_trim.pickle')
products.to_pickle('../processed_data/feature_products.pickle')


