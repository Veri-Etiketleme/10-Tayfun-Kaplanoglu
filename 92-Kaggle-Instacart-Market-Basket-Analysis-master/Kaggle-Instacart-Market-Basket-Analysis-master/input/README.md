### Dateset
The dataset is anonymized and contains a sample of over 3 million grocery orders from more than 200,000 Instacart users. For more information, see the [blog post](https://tech.instacart.com/3-million-instacart-orders-open-sourced-d40d29ead6f2). The dataset can be downloaded in [kaggle](https://www.kaggle.com/c/instacart-market-basket-analysis/data)

### File Description
- **aisles.csv**

```
aisle_id,aisle  
 1,prepared soups salads  
 2,specialty cheeses  
 3,energy granola bars  
 ...

```
- **departments.csv**

```
department_id,department  
 1,frozen  
 2,other  
 3,bakery  
 ...

```
- **order_products__*.csv**

```
order_id,product_id,add_to_cart_order,reordered  
 1,49302,1,1  
 1,11109,2,1  
 1,10246,3,0  
 ... 

```
- **orders.csv**

```
order_id,user_id,eval_set,order_number,order_dow,order_hour_of_day,days_since_prior_order  
 2539329,1,prior,1,2,08,  
 2398795,1,prior,2,3,07,15.0  
 473747,1,prior,3,3,12,21.0  
 ...

```
- **products.csv**

```
product_id,product_name,aisle_id,department_id
 1,Chocolate Sandwich Cookies,61,19  
 2,All-Seasons Salt,104,13  
 3,Robust Golden Unsweetened Oolong Tea,94,7  
 ...

```

- **sample_submission.csv**

```
order_id,products
17,39276  
34,39276  
137,39276  
...

```