import pandas as pd
from feature_df_helper import features_df
import pickle

orders_trim = pd.read_pickle('../processed_data/orders_trim.pickle')
train = pd.read_csv('../input/order_products__train.csv')
users = pd.read_pickle('../processed_data/feature_users.pickle')
products = pd.read_pickle('../processed_data/feature_products.pickle')
userXproduct = pd.read_pickle('../processed_data/feature_userXproduct.pickle')
aisle = pd.read_pickle('../processed_data/feature_aisle.pickle')
depart = pd.read_pickle('../processed_data/feature_depart.pickle')
userXaisle = pd.read_pickle('../processed_data/feature_userXaisle.pickle')
userXdepart = pd.read_pickle('../processed_data/feature_userXdepart.pickle')

print('split orders : train, test')
test_orders = orders_trim[orders_trim.eval_set == 'test']
train_orders = orders_trim[orders_trim.eval_set == 'train']


train.set_index(['order_id', 'product_id'], inplace=True, drop=False)


df_train, labels_train = features_df(train_orders, train, users, orders_trim, products, aisle, depart, \
                                     userXproduct, userXaisle, userXdepart, labels_given=True)

df_test, _ = features_df(test_orders, train, users, orders_trim, products, aisle, depart, \
                                     userXproduct, userXaisle, userXdepart, labels_given=False)

# save df to pickle
with open('../processed_data/df_train.pickle', 'wb') as handle:
    pickle.dump(df_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../processed_data/labels_train.pickle', 'wb') as handle:
    pickle.dump(labels_train, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('../processed_data/df_test.pickle', 'wb') as handle:
    pickle.dump(df_test, handle, protocol=pickle.HIGHEST_PROTOCOL)