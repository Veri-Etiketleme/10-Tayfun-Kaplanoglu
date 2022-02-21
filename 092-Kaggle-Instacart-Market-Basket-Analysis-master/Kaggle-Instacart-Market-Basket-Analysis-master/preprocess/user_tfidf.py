# reference post: https://www.kaggle.com/tedchang0102/using-tfidf-as-user-s-affinity
"""
for each use_id, get a list of products the user has reordered. Apply tfidf to the product list
and generate 2 components associated to the user

"""


import numpy as np
import pandas as pd
import scipy.sparse as ssp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA


print('User reordered products rfidf features')

# Load the data
orders = pd.read_csv("../input/orders.csv")
train_orders = pd.read_csv("../input/order_products__train.csv")
prior_orders = pd.read_csv("../input/order_products__prior.csv")
products = pd.read_csv("../input/products.csv").set_index('product_id')


# get the products that have been reordered
prior_orders = prior_orders[prior_orders.reordered==1]
prior_ord = pd.merge(prior_orders,orders,on='order_id',how='left')
products = products.reset_index()


# merge prior order to product
prior_ord = pd.merge(prior_ord,products,on='product_id',how='left')


# df with user_id and a list of products the user reordered
prior_ord["product_name"] = prior_ord["product_name"].astype(str)
prior_ord = prior_ord.groupby("user_id").apply(lambda order: order['product_name'].tolist())
prior_ord = prior_ord.reset_index()
prior_ord.columns = ['user_id','product_set']
prior_ord.product_set = prior_ord.product_set.astype(str)

# set up tfidf
tfidf = TfidfVectorizer(min_df=5, max_features=1000
                        , strip_accents='unicode',lowercase =True,
analyzer='word', token_pattern=r'\w+', use_idf=True,
smooth_idf=True, sublinear_tf=True, stop_words = 'english')
tfidf.fit(prior_ord['product_set'])

# apply tfidf transform to the product list or each order and output 2 principal components
text = tfidf.transform(prior_ord['product_set'])
svd = TruncatedSVD(n_components=2)
text = svd.fit_transform(text)
text = pd.DataFrame(text)
text.columns = ['pf_0','pf_1']
text['user_id'] = prior_ord.user_id

# save the two components with the user_id
text.set_index('user_id').to_csv('../processed_data/user_rfidf.csv', index=False)