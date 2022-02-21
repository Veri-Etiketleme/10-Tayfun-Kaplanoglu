import pandas as pd
import numpy as np

def features_df(selected_orders, train, users, orders_trim, products, aisle, depart, userXproduct, userXaisle, userXdepart, labels_given=False):
    print('build candidate list')
    order_list = []
    product_list = []
    labels = []
    i = 0
    train_index_lookup = dict().fromkeys(train.index.values)
    for row in selected_orders.itertuples():
        i += 1
        if i % 5000 == 0: print('order row', i)
        order_id = row.order_id
        user_id = row.user_id
        user_products = users.all_products[user_id]
        product_list += user_products
        order_list += [order_id] * len(user_products)
        if labels_given:
            labels += [(order_id, product) in train_index_lookup for product in user_products]

    df = pd.DataFrame({'order_id': order_list, 'product_id': product_list}, dtype=np.int32)
    labels = np.array(labels, dtype=np.int8)
    del order_list
    del product_list

    print('user related features')
    df['user_id'] = df.order_id.map(orders_trim.user_id)
    df['user_total_orders'] = df.user_id.map(users.nb_orders)
    df['user_total_items'] = df.user_id.map(users.total_items)
    df['user_total_distinct_items'] = df.user_id.map(users.total_distinct_items)
    df['user_average_days_between_orders'] = df.user_id.map(users.average_days_between_orders)
    df['user_average_basket'] = df.user_id.map(users.average_basket)
    df['user_avg_order_dow'] = df.user_id.map(users.avg_order_dow)
    df['user_avg_order_hour_of_day'] = df.user_id.map(users.avg_order_hour_of_day)
    df['user_last_product_count'] = df.user_id.map(users.last_product_count)
    df['user_last_vs_previous_basket'] = df.user_id.map(users.last_vs_previous_basket)
    df['user_distinct_ratio'] = df.user_id.map(users.last_product_count)
    df['user_mode_order_dow'] = df.user_id.map(users.mode_order_dow)
    df['user_mode_order_hour_of_day'] = df.user_id.map(users.mode_order_hour_of_day)
    df['user_nb_weekend_order'] = df.user_id.map(users.nb_weekend_order)
    df['user_nb_weekly_order'] = df.user_id.map(users.nb_weekly_order)
    df['user_weekend_ratio'] = df.user_id.map(users.weekend_ratio)
    df['user_weekly_ratio'] = df.user_id.map(users.weekly_ratio)
    df['user_first_avg_days_between_orders'] = df.user_id.map(users.first_avg_days_between_orders)
    df['user_last_avg_days_between_orders'] = df.user_id.map(users.last_avg_days_between_orders)
    df['user_last_first_avg_days_ratio'] = df.user_id.map(users.last_first_avg_days_ratio)
    df['user_first_two_items_number'] = df.user_id.map(users.first_two_items_number)
    df['user_last_two_items_number'] = df.user_id.map(users.last_two_items_number)
    df['user_first_two_reorder_number'] = df.user_id.map(users.first_two_reorder_number)
    df['user_last_two_reorder_number'] = df.user_id.map(users.last_two_reorder_number)
    df['user_first_last_number_ratio'] = df.user_id.map(users.first_last_number_ratio)
    df['user_first_last_reorder_ratio'] = df.user_id.map(users.first_last_reorder_ratio)
    df['user_diff_hour_since_last'] = df.user_id.map(users.diff_hour_since_last)
    df['user_last2_all_reorder_ratio'] = df.user_id.map(users.last2_all_reorder_ratio)
    df['user_last_reorder_count'] = df.user_id.map(users.last_reorder_count)
    df['user_last_reorder_rate'] = df.user_id.map(users.last_reorder_rate)
    df['user_pf_0'] = df.user_id.map(users.pf_0)
    df['user_pf_1'] = df.user_id.map(users.pf_1)
    df['user_tenure'] = df.user_id.map(users.tenure)
    df['user_avg_nb_tenure'] = df.user_id.map(users.avg_nb_tenure)
    df['user_nb_days_is_30'] = df.user_id.map(users.user_nb_days_is_30)
    df['user_nb_days_is_30_ratio'] = df.user_id.map(users.user_nb_days_is_30_ratio)

    print('order related features')
    df['order_dow'] = df.order_id.map(orders_trim.order_dow)
    df['order_hour_of_day'] = df.order_id.map(orders_trim.order_hour_of_day)
    df['order_days_since_prior_order'] = df.order_id.map(orders_trim.days_since_prior_order)
    df['order_days_since_ratio'] = df.order_days_since_prior_order / df.user_average_days_between_orders
    df['order_weekend'] = df.order_id.map(orders_trim.weekend)
    df['order_weekly'] = df.order_id.map(orders_trim.weekly)
    df['order_days_is_30'] = df.order_id.map(orders_trim.days_is_30)
    df['order_number'] = df.order_id.map(orders_trim.order_number)

    print('product related features')
    df['product_aisle_id'] = df.product_id.map(products.aisle_id)
    df['product_department_id'] = df.product_id.map(products.department_id)
    df['product_orders'] = df.product_id.map(products.orders)
    df['product_reorders'] = df.product_id.map(products.reorders)
    df['product_reorder_rate'] = df.product_id.map(products.reorder_rate)
    df['product_order_in_cart'] = df.product_id.map(products.order_in_cart)
    #df['product_key'] = df.product_id.map(products.product_key)
    df['product_avg_hour_of_day'] = df.product_id.map(products.prod_avg_hour_of_day)
    df['product_avg_days_since'] = df.product_id.map(products.prod_avg_days_since)
    df['product_avg_dow'] = df.product_id.map(products.prod_avg_dow)
    df['product_avg_order_number'] = df.product_id.map(products.prod_avg_order_number)
    df['product_nb_weekend_order'] = df.product_id.map(products.prod_nb_weekend_order)
    df['product_nb_weekly_order'] = df.product_id.map(products.prod_nb_weekly_order)
    df['product_weekend_ratio'] = df.product_id.map(products.prod_weekend_ratio)
    df['product_weekly_ratio'] = df.product_id.map(products.prod_weekly_ratio)
    df['product_dimension_1'] = df.product_id.map(products.dimension_1)
    df['product_dimension_2'] = df.product_id.map(products.dimension_2)
    df['product_dimension_3'] = df.product_id.map(products.dimension_3)

    df['product_std_days_since'] = df.product_id.map(products.prod_std_days_since)
    df['product_nb_days_is_30'] = df.product_id.map(products.prod_nb_days_is_30)
    df['product_days_30_ratio'] = df.product_id.map(products.prod_days_30_ratio)

    print('aisle related features')
    df['as_orders'] = df.product_aisle_id.map(aisle.as_orders)
    df['as_reorders'] = df.product_aisle_id.map(aisle.as_reorders)
    df['as_reorder_rate'] = df.product_aisle_id.map(aisle.as_reorder_rate)

    df['as_avg_days_since'] = df.product_aisle_id.map(aisle.as_avg_days_since)
    df['as_unique_user'] = df.product_aisle_id.map(aisle.as_unique_user)
    df['as_order_per_unique_usere'] = df.product_aisle_id.map(aisle.as_order_per_unique_user)
    df['as_avg_add_to_cart'] = df.product_aisle_id.map(aisle.as_avg_add_to_cart)
    df['as_unique_prod'] = df.product_aisle_id.map(aisle.as_unique_prod)

    print('department related features')
    df['dp_orders'] = df.product_department_id.map(depart.dp_orders)
    df['dp_reorders'] = df.product_department_id.map(depart.dp_reorders)
    df['dp_reorder_rate'] = df.product_department_id.map(depart.dp_reorder_rate)

    df['dp_avg_days_since'] = df.product_department_id.map(depart.dp_avg_days_since)
    df['dp_unique_user'] = df.product_department_id.map(depart.dp_unique_user)
    df['dp_order_per_unique_user'] = df.product_department_id.map(depart.dp_order_per_unique_user)
    df['dp_avg_add_to_cart'] = df.product_department_id.map(depart.dp_avg_add_to_cart)
    df['dp_unique_prods'] = df.product_department_id.map(depart.dp_unique_prod)

    print('user_X_product related features')
    df['z'] = df.user_id * 100000 + df.product_id

    df['UP_orders'] = df.z.map(userXproduct.user_product_count)
    df['UP_last_order_id'] = df.z.map(userXproduct.last_order_id)
    df['UP_average_pos_in_cart'] = df.z.map(userXproduct.add_to_cart_order_mean)
    df['UP_reorder_rate'] = (df.UP_orders / df.user_total_orders)
    df['UP_order_number_min'] = df.z.map(userXproduct.order_number_min)
    df['UP_order_number_max'] = df.z.map(userXproduct.order_number_max)
    df['UP_orders_since_last'] = df.user_total_orders - df.UP_order_number_max
    df['UP_orders_since_first'] = df.user_total_orders - df.UP_order_number_min
    df['UP_order_rate_since_first_order'] = df.UP_orders / (df.user_total_orders - df.UP_order_number_min + 1)
    df['UP_delta_hour_vs_last'] = abs(
        df.order_hour_of_day - df.UP_last_order_id.map(orders_trim.order_hour_of_day)).map(lambda x: min(x, 24 - x))
    df['UP_first_order_rate'] = df.z.map(userXproduct.order_number_min) / df.user_total_orders
    df['UP_last_order_rate'] = df.z.map(userXproduct.order_number_max) / df.user_total_orders
    # df['UP_order_since_last_order'] = df.user

    df['UP_order_dow_mean'] = df.z.map(userXproduct.order_dow_mean)
    df['UP_order_hour_of_day_mean'] = df.z.map(userXproduct.order_hour_of_day_mean)
    df['UP_days_since_prior_order_mean'] = df.z.map(userXproduct.days_since_prior_order_mean)

    df['UP_order_rate_since_first'] = df.z.map(userXproduct.order_rate_since_first)
    df['UP_weekend_number'] = df.z.map(userXproduct.up_weekend_number)
    df['UP_weekly_number'] = df.z.map(userXproduct.up_weekly_number)
    df['UP_weekend_ratio'] = df.z.map(userXproduct.up_weekend_ratio)
    df['UP_weekly_ratio'] = df.z.map(userXproduct.up_weekly_ratio)
    df['UP_first_two_reordered'] = df.z.map(userXproduct.up_first_two_reordered)
    df['UP_last_two_reordered'] = df.z.map(userXproduct.up_last_two_reordered)
    df['UP_days_is_30_sum'] = df.z.map(userXproduct.days_is_30_sum)
    df['UP_days_since_prior_order_std'] = df.z.map(userXproduct.days_since_prior_order_std)

    print('user_X_aisle related features')
    df['ua'] = df.product_aisle_id + df.user_id * 100000
    # df.drop(['user_id'], axis=1, inplace=True)
    df['ua_orders'] = df.ua.map(userXaisle.ua_orders)
    df['ua_reorder'] = df.ua.map(userXaisle.ua_reorder)
    df['ua_reorder_rate'] = df.ua.map(userXaisle.ua_reorder_rate)
    df['ua_avg_days_since'] = df.ua.map(userXaisle.ua_avg_days_since)

    print('user_X_depart related features')
    df['ud'] = df.product_department_id + df.user_id * 100000
    df.drop(['user_id'], axis=1, inplace=True)
    df['ud_orders'] = df.ud.map(userXdepart.ud_orders)
    df['ud_reorder'] = df.ud.map(userXdepart.ud_reorder)
    df['ud_reorder_rate'] = df.ud.map(userXdepart.ud_reorder_rate)
    df['ud_avg_days_since'] = df.ud.map(userXdepart.ud_avg_days_since)

    df.drop(['UP_last_order_id', 'z', 'ua', 'ud'], axis=1, inplace=True)
    # print(df.dtypes)
    # print(df.memory_usage())
    return (df, labels)
