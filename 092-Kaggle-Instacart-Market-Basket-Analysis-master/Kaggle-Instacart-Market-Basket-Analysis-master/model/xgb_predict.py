import xgboost as xgb
import pickle
import numpy as np

# Load the model and test data
with open('../processed_data/df_test.pickle', 'rb') as handle:
    df_test = pickle.load(handle)

with open('xgb_model.pickle', 'rb') as handle:
    bst = pickle.load(handle)

f_to_use = ['user_total_orders', 'user_total_items', 'user_total_distinct_items', 'user_last_product_count',
            'user_last_vs_previous_basket', 'user_distinct_ratio', 'user_mode_order_dow', 'user_mode_order_hour_of_day',
            'user_nb_weekend_order', 'user_nb_weekly_order', 'user_weekend_ratio', 'user_weekly_ratio',
            'user_last_avg_days_between_orders', 'user_last_first_avg_days_ratio', 'user_last_two_items_number',
            'user_first_two_reorder_number', 'user_last_two_reorder_number', 'user_first_last_number_ratio',
            'user_first_last_reorder_ratio', 'user_diff_hour_since_last', 'user_last2_all_reorder_ratio',
            'user_last_reorder_count', 'user_last_reorder_rate', 'user_tenure', 'user_avg_nb_tenure',
            'user_nb_days_is_30', 'user_nb_days_is_30_ratio', 'order_hour_of_day', 'order_days_since_prior_order',
            'order_days_since_ratio', 'order_weekend', 'order_weekly', 'order_days_is_30', 'order_number',
            'product_orders', 'product_reorder_rate', 'product_avg_days_since', 'product_avg_dow',
            'product_avg_order_number', 'product_nb_weekend_order', 'product_nb_weekly_order', 'product_weekend_ratio',
            'product_weekly_ratio', 'product_dimension_1', 'product_dimension_2', 'product_dimension_3',
            'product_std_days_since', 'product_nb_days_is_30', 'product_days_30_ratio', 'as_orders', 'as_reorders',
            'as_reorder_rate', 'as_avg_days_since', 'as_unique_user', 'as_avg_add_to_cart', 'as_unique_prod',
            'dp_orders', 'dp_reorders', 'dp_reorder_rate', 'dp_avg_days_since', 'dp_order_per_unique_user',
            'dp_avg_add_to_cart', 'dp_unique_prods', 'UP_average_pos_in_cart', 'UP_reorder_rate', 'UP_order_number_min',
            'UP_order_number_max', 'UP_orders_since_last', 'UP_orders_since_first', 'UP_order_rate_since_first_order',
            'UP_delta_hour_vs_last', 'UP_order_dow_mean', 'UP_order_hour_of_day_mean', 'UP_days_since_prior_order_mean',
            'UP_weekend_number', 'UP_weekly_number', 'UP_weekend_ratio', 'UP_weekly_ratio', 'UP_first_two_reordered',
            'UP_last_two_reordered', 'UP_days_is_30_sum', 'UP_days_since_prior_order_std', 'ua_orders', 'ua_reorder',
            'ua_reorder_rate', 'ud_orders', 'ud_reorder', 'ud_reorder_rate', 'ud_avg_days_since', 'product_aisle_id',
            'product_department_id', 'order_dow']

# Split the test_df into 20 chunks
print('prdicting the test data')
test_res = []
# i = 0
for test_features_chunk in np.array_split(df_test, 20):
    # print(i)
    # i += 1
    d_test = xgb.DMatrix(test_features_chunk[f_to_use])
    test_res.append(bst.predict(d_test))

final_res = np.concatenate(test_res)

df_test['pred'] = final_res

df_test[['order_id', 'product_id', 'pred']].to_pickle('../processed_data/pred_df_test.pickle')
# print(df_test[['order_id', 'product_id', 'pred']].head())
