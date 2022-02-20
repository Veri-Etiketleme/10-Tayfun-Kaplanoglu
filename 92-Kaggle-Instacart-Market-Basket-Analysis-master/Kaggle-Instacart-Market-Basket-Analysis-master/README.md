## Instacart Market Basket Analysis

Solution to the Kaggle Competition: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis) 

### Summary Report ([pdf](summary_report.pdf))

### Dataset
The dataset is an open-source dataset provided by Instacart.

 >This anonymized dataset contains a sample of over 3 million grocery orders from more than 200,000 Instacart users. For each user, we provide between 4 and 100 of their orders, with the sequence of products purchased in each order. 

Below is the full data schema ([source](https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b))

 > `orders` (3.4m rows, 206k users):
 > * `order_id`: order identifier
 > * `user_id`: customer identifier
 > * `eval_set`: which evaluation set this order belongs in (see `SET` described below)
 > * `order_number`: the order sequence number for this user (1 = first, n = nth)
 > * `order_dow`: the day of the week the order was placed on
 > * `order_hour_of_day`: the hour of the day the order was placed on
 > * `days_since_prior`: days since the last order, capped at 30 (with NAs for `order_number` = 1)
 >
 > `products` (50k rows):
 > * `product_id`: product identifier
 > * `product_name`: name of the product
 > * `aisle_id`: foreign key
 > * `department_id`: foreign key
 >
 > `aisles` (134 rows):
 > * `aisle_id`: aisle identifier
 > * `aisle`: the name of the aisle
 >
 > `deptartments` (21 rows):
 > * `department_id`: department identifier
 > * `department`: the name of the department
 >
 > `order_products__SET` (30m+ rows):
 > * `order_id`: foreign key
 > * `product_id`: foreign key
 > * `add_to_cart_order`: order in which each product was added to cart
 > * `reordered`: 1 if this product has been ordered by this user in the past, 0 otherwise
 >
 > where `SET` is one of the four following evaluation sets (`eval_set` in `orders`):
 > * `"prior"`: orders prior to that users most recent order (~3.2m orders)
 > * `"train"`: training data supplied to participants (~131k orders)
 > * `"test"`: test data reserved for machine learning competitions (~75k orders)


### Problem Statement
Use the anonymized data on customer orders over time to predict which previous purchased products will be in a user’s next order. The task can be formulated as a binary prediction: given a user, a product(this product has been purchased by the user in the previous orders) and the user’s prior purchase history, predict whether or not the given product will be reordered in the user’s next order. The output format will be: user’s last order id and a list of products that will be reordered. 

### Evaluation Metric
F1 score between the set of predicted reordered products and the set of true reordered products

### The Approach
- **Exploratory Data Analysis**: Further explore the dataset to see if we have any missing data or outliers in the orders and user dataset.
- **Sampling**: It’s a huge dataset and the memory may be an issue when running the machine learning model. I sampled 20% of the training set for machine learning model most of the time. ([code](./preprocess/sampling_cv.py))
- **Feature Engineering**: feature engineering is the key in the competition. User-related, order-related, product-related features are extracted from the dataset.([user-features](./feature/feature_users.py), [aisle/depart-features](./feature/feature_aisle_depart.py), [product-features](./feature/feature_products.py), [userXaisle/depart-features](./feature/feature_userXaisleXdepart.py), [userXproduct-features](./feature/feature_userXproduct.py),) Other more complicated features like user-product interaction features and NLP are applied to generate new features as well.([tfidf-features](./preprocess/user_tfidf.py), [Word2Vec-features](./preprocess/product_Word2Vec.py))
- **Machine Learning Model**: I built a user-product machine learning model and apply logic regression to output the re-order probability for each product.  There are many features involved in the model, popular tree-based model-xgboost is used.([code](.model))
- **F1-optimization**: The output of the machine learning model is a list of products the user purchased and the re-order probability for each product. I apply an algorithm to maximize the f1 score for the product list selected given their probabilities to be re-ordered.([code](./f1_optimal/f1_optimized.py))
- **Model Stacking**: It’s a kaggle open competition and some kagglers shared their model and predictions in the forum. In the final stage, I used some stacking technologies to combine my model and other well-performed [shared model](https://www.kaggle.com/c/instacart-market-basket-analysis/discussion/37697) results for final submission. ([code](./f1_optimal/median_stacking.py))

### Results
Ranked 230/2630 (top 9%)

### Requirements
16 GB RAM, Python 3.4 with the following Python packages installed:

- numpy == 1.12.1
- pandas == 0.20.2
- xgboost == 0.6
- scikit-learn == 0.12.2
- gensim == 2.2.0

