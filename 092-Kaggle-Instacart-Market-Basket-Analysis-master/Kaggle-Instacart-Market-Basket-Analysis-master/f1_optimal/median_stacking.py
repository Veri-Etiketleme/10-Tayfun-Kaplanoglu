import pandas as pd
import pickle

with open('../processed_data/prediction_lgbm.pkl', 'rb') as f:
    data_lgbm = pickle.load(f)

data_lgbm.columns = ['order_id', 'pred', 'product_id']

with open('../processed_data/prediction_arboretum.pkl', 'rb') as f:
    data_arb = pickle.load(f)

data_arb.columns = ['order_id', 'pred', 'product_id']

with open('../processed_data/pred_df_test.pickle', 'rb') as f:
    data_org = pickle.load(f)

data_org.columns = ['order_id', 'pred', 'product_id']

df = pd.merge(data_lgbm, data_arb, on=['order_id', 'product_id'])
df = df.merge(data_org, on=['order_id', 'product_id'])
df.loc[:, 'pred_mean'] = df[['pred_x', 'pred_y', 'pred']].median(axis=1)
df = df[['order_id', 'product_id', 'pred_mean']]
df.columns = ['order_id', 'product_id', 'pred']
df.to_pickle('../processed_data/stacking.pickle')