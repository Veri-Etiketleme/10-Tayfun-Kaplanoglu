import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
from f1_optimizer_helper import F1Optimizer
import operator
from pandas import Series

# Load the test data with predicted probability
with open('../processed_data/pred_df_test.pickle', 'rb') as handle:
    df_test = pickle.load(handle)

# transform the df to order_id and a list of product re-order probability
order_prod_list = df_test.groupby('order_id')['pred'].apply(list).reset_index()

# get the top n products to select from each order based on f1_opt
a = F1Optimizer()
order_prod_list = order_prod_list.head()
order_prod_list['opt'] = order_prod_list['pred'].apply(lambda x:a.maximize_expectation(x))

# get a list of orders based on the pred, top_n, and if None should be included
group_test_prod = df_test.groupby(['order_id'])

def agg_optimized_test_result(group):
    return pd.Series(group['pred'].values, index=group.product_id).to_dict()

agg_test_prod = group_test_prod.apply(agg_optimized_test_result).reset_index().rename(columns={0: 'prod_pred_dict'})
agg_test_prod = agg_test_prod.merge(order_prod_list.reset_index(), on='order_id', how='left')


def func(row):
    n_items, include_none, _ = row['opt']
    res_list = sorted(row['prod_pred_dict'].items(), key=operator.itemgetter(1), reverse=True)
    #print (res_list[n_items-1 : n_items + 1], n_items)
    res_list = list(map(lambda item: item[0], res_list[:n_items]))
    if include_none or not res_list:
        res_list.append('None')
    return res_list

final_result = pd.DataFrame()
final_result['order_id'] = agg_test_prod['order_id']
final_result['optimized_result'] = agg_test_prod.apply(func, axis=1)
final_result['products'] = final_result['optimized_result'].apply(lambda x:" ".join(str(i) for i in x))

final_result[['order_id', 'products']].to_csv('../output/summit.csv', index=False)