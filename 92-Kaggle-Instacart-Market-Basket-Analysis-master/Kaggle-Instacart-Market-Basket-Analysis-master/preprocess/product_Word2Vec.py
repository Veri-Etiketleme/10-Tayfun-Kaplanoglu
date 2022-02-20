# reference post: https://omarito.me/word2vec-product-recommendations/
#                 https://www.kaggle.com/omarito/word2vec-for-products-analysis-0-01-lb
"""
Word2Vec can transform the word in the vocabulary into a meaningful vector representation instead
of extremely sparse bow representation.

"""

# Load libraries
import pandas as pd
import numpy as np
import gensim
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

print('Product Word2Vec features')

# Load data
train_orders = pd.read_csv("../input/order_products__train.csv")
prior_orders = pd.read_csv("../input/order_products__prior.csv")
products = pd.read_csv("../input/products.csv").set_index('product_id')

# Turn the product id into a string
train_orders["product_id"] = train_orders["product_id"].astype(str)
prior_orders["product_id"] = prior_orders["product_id"].astype(str)

# Extract the ordered products in each order
train_products = train_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())
prior_products = prior_orders.groupby("order_id").apply(lambda order: order['product_id'].tolist())

# Create the final sentences
sentences = prior_products.append(train_products)
longest = np.max(sentences.apply(len))
sentences = sentences.values

# Train Word2Vec model
model = gensim.models.Word2Vec(sentences, size=100, window=longest, min_count=2, workers=4)

#
vec_product = products.reset_index()
vec_product["product_id_str"] = vec_product["product_id"].astype(str)
vec_product['vec_array'] = vec_product['product_id_str'].apply(lambda x: model[x] if x in list(model.wv.vocab.keys()) else np.nan)

# df with product id and a vector associated with the id
id_vec = vec_product[['product_id', 'vec_array']].dropna()

# Apply PCA to generate 3 components
id_vec_split = pd.DataFrame([x for x in id_vec.vec_array])
pca = PCA(n_components=3)
pca.fit(id_vec_split)

reduced_data_df = pd.DataFrame(pca.transform(id_vec_split), columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])

combine_df = pd.concat([id_vec, reduced_data_df], axis=1)[['product_id', 'Dimension 1', 'Dimension 2', 'Dimension 3']]
combine_vec_product = vec_product.merge(combine_df, on='product_id', how='left').drop('vec_array', 1)

combine_vec_product.to_csv('../processed_data/products_PCA.csv', index=False)

