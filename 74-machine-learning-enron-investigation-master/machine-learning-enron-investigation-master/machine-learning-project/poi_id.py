#!/usr/bin/python
import matplotlib.pyplot as plt
import pprint as pp
import numpy as np
import operator
import pandas as pd
from time import time

from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import *
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.naive_bayes import GaussianNB
from sklearn import grid_search

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
#lets verify if data is as expected with given features
print 'Number of features: ', len(data_dict[data_dict.keys()[0]].keys())
pp.pprint(data_dict[data_dict.keys()[0]].keys())
print '\n'

#verify total number of enron employees and POIs in data
print 'Total Number of Employees: ', len(data_dict)
poi_count = 0
for key, value in data_dict.items():
    if value['poi'] == True:
        poi_count += 1
    
print 'Total Number of POIs: ', poi_count

#lets verify the data quality
print '*** Data Quality ***'
print 'Feature, [Total Available, Percentage, POIs]'
features_data = {}  # count of number of features having data
features = data_dict[data_dict.keys()[0]].keys()
for feature in features:
    features_data[feature] = [0, 0, 0]
for emp_features in data_dict.values():
    for feature in features:
        if feature in emp_features.keys():
            if feature == 'email_address': 
                if emp_features[feature] != '':
                    features_data[feature][0] += 1
                    if emp_features['poi'] == True:
                        features_data[feature][2] += 1
            elif type(emp_features[feature]) != type('str'):
                features_data[feature][0] += 1
                if emp_features['poi'] == True:
                    features_data[feature][2] += 1
#calculate percentage of employees having data                
for key, value in features_data.items():
    features_data[key][1] = (value[0]*100/len(data_dict))

pp.pprint(sorted(features_data.items(), key=operator.itemgetter(1)))
print '\n'

# Histogram on total payment to check any behaviour for POIs
poi_total_payments = []
non_poi_total_payments = []
for key, value in data_dict.items():
    if value['poi'] == True:
        poi_total_payments.append(value['total_payments'])
    else:
        if type(value['total_payments']) <> type('str'):
            non_poi_total_payments.append(value['total_payments'])

plt.hist(poi_total_payments, label='POI')
plt.hist(non_poi_total_payments, color='r', alpha=0.2, label='non-POI')
plt.title("POIs/non-POIs based on Total Payments")
plt.xlabel("Total Payment")
plt.ylabel("number of POIs/non-POIs")
plt.legend()
#plt.show()

#Calculate IQR to find outliers
payments = []
for key, value in data_dict.items():
    if type(value['total_payments']) <> type('str'): 
        payments.append(value['total_payments'])
                
q75, q25 = np.percentile(payments, [75 ,25])
print 'Total_Payment IQR-Q1: ', q25, 'IQR-Q3: ',q75, '\n'

#Lets run the histogram again for payments less than Q75
#Do not ignore data under IQR-Q1 now. Lets check how the plot looks at lowest values
poi_total_payments = []
ol_poi_total_payments = []  #outlier data
non_poi_total_payments = []
ol_non_poi_total_payments = []   #outlier data
for key, value in data_dict.items():
    if value['poi'] == True:
        if value['total_payments'] < q75:
            poi_total_payments.append(value['total_payments'])
        else:
            ol_poi_total_payments.append(value['total_payments'])
    else:
        if type(value['total_payments']) <> type('str'):
            if value['total_payments'] < q75:
                non_poi_total_payments.append(value['total_payments'])
            else:
                ol_non_poi_total_payments.append(value['total_payments'])
                
plt.hist(poi_total_payments, label='POI', bins=30)
plt.hist(non_poi_total_payments, color='r', alpha=0.2, label='non-POI', bins=20)
plt.title("POIs/non-POIs based on Total Payments")
plt.xlabel("Total Payment")
plt.ylabel("number of POIs/non-POIs")
plt.legend()
#plt.show()

print '\n', '*** Employees by Total_Payments  ***'
print 'number of POIs under IQR-Q3: ', len(poi_total_payments)
print 'number of non-POIs under IQR-Q3: ', len(non_poi_total_payments)

print '\n', '*** Outliers on Total_payments ***'
plt.hist(ol_poi_total_payments, label='POI', bins=30)
plt.hist(ol_non_poi_total_payments, color='r', alpha=0.2, label='non-POI', bins=20)
plt.title("POIs/non-POIs based on Total Payments")
plt.xlabel("Total Payment")
plt.ylabel("number of POIs/non-POIs")
plt.legend()
#plt.show()

print 'number of POI in Outliers: ', len(ol_poi_total_payments)
print 'number of non-POIs in Outliers: ', len(ol_non_poi_total_payments)
print '\n'

# Lets see outliers with total payment > 100 mm
print '*** outliers with total payment > 100 mm ***'
for key, value in data_dict.items():
    if type(value['total_payments']) <> type('str') and value['total_payments'] >= 1e8:
        print key
        pp.pprint(value)
        print '\n'
#Delete LAY KENNETH L and TOTAL
del data_dict['LAY KENNETH L']
del data_dict['TOTAL']
#check number of employees to make sure only 2 are removed
print 'Number of Employees after Outlier removed: ', len(data_dict), '\n'	

### create new features
### We do have total number of emails and POI emails (from/to), I would rather use 
### percentage of emails from/to POIs to analyse behaviour. 
### new features are: fraction_to_poi_email,fraction_from_poi_email
### calculate fraction (0 to 1) for percentage

def dict_to_list(key,normalizer):
    new_list=[]

    for i in data_dict:
        if data_dict[i][key]=="NaN" or data_dict[i][normalizer]=="NaN":
            new_list.append(0.)
        elif data_dict[i][key]>=0:
            new_list.append(float(data_dict[i][key])/float(data_dict[i][normalizer]))
    return new_list

### create two lists of new features
fraction_from_poi_email=dict_to_list("from_poi_to_this_person","to_messages")
fraction_to_poi_email=dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into data_dict
count=0
for i in data_dict:
    data_dict[i]["fraction_from_poi_email"]=fraction_from_poi_email[count]
    data_dict[i]["fraction_to_poi_email"]=fraction_to_poi_email[count]
    count +=1

#Lets plot histogram by new fraction of emails from/to POI
features_list = ["poi", "fraction_from_poi_email", "fraction_to_poi_email"]    
    ### store to my_dataset for easy export below
my_dataset = data_dict

### these two lines extract the features specified in features_list
### and extract them from data_dict, returning a numpy array
data = featureFormat(my_dataset, features_list)

### plot new features
for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0] == 1:
        pl_poi = plt.scatter(from_poi, to_poi, color="r", marker="*")
    else:
        pl_nonPOI = plt.scatter( from_poi, to_poi )
plt.xlabel("portion of emails this person gets from poi")
plt.ylabel("portion of emails this person sends to poi")
plt.legend((pl_poi, pl_nonPOI), ('POI', 'non-POI'))
#plt.show()

#### Run supervised learning algorithm
def train_test_data(features_):
    data = featureFormat(my_dataset, features_)

    ### split into labels and features (this line assumes that the first
    ### feature in the array is the label, which is why "poi" must always
    ### be first in features_list
    labels, features = targetFeatureSplit(data)

    ### split data into training and testing datasets
    return cross_validation.train_test_split(features, labels, test_size=0.30, random_state=42)

def test_algorithm(clf, features_tr, features_ts):
    t0 = time()
    clf.fit(features_tr,labels_train)
    score = clf.score(features_ts,labels_test)
    pred= clf.predict(features_ts)
    run_time = round(time()-t0, 3)
    return {'Accuracy':round(score,2), 'Precision':round(precision_score(labels_test, pred), 2), 
            'Recall': round(recall_score(labels_test, pred), 2),
            'F1_Score': round(f1_score(labels_test, pred), 2), 'Duration_sec': run_time}

features_li = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'total_payments', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'shared_receipt_with_poi', 'restricted_stock']
features_train, features_test, labels_train, labels_test = train_test_data(features_li)
print 'Make sure we have approximately equal percentage of POIs in Train and Test out of 18 POIs.'
print 'Test labels: ', labels_test, '\n'    
clf0 = DecisionTreeClassifier()
print '*** Decesiontree Algorithm - features with atleast 50% having data***'
print test_algorithm(clf0,features_train,features_test), '\n'

features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']
features_train, features_test, labels_train, labels_test = train_test_data(features_list)
    
clf1 = DecisionTreeClassifier()
print '*** Decesiontree Algorithm with all features ***'
print test_algorithm(clf1,features_train,features_test), '\n'

importances = clf1.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(16):
    print "\t {} {} ({})".format(i+1,features_list[i+1],importances[indices[i]])
	
#Lets try selecting top percentile features and check if we can get same or better result
def select_features(clf, perc_fea):
    selector = SelectPercentile(f_classif, percentile=perc_fea)
    selector.fit(features_train,labels_train)
    features_train_s = selector.transform(features_train)
    features_test_s = selector.transform(features_test)
    metrics = test_algorithm(clf,features_train_s,features_test_s)
    metrics['Percentile'] = perc_fea
    print metrics, '\n'

selector = SelectPercentile(f_classif, percentile=100)
selector.fit(features_train,labels_train)
print '\n', '** SelectPercentile Feature scores **'
indices = np.argsort(selector.scores_)[::-1]
for i in range(16):
    print "\t {} {} ({})".format(i+1,features_list[i+1],selector.scores_[indices[i]])

print '\n', '*** Decesiontree Algorithm with Select percentile features ***'
percents = [90, 80, 70, 60, 50, 40, 30, 20, 10]
for percent in percents:
    clf2 = DecisionTreeClassifier()
    select_features(clf2, percent)
	
print '\n', '*** Naive Bayes Algorithm with percentile features ***'
percents = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10]
for percent in percents:
    clf3 = GaussianNB()
    select_features(clf3, percent)

#Lets try manually selecting payment features
features_manual = ["poi", "salary", "bonus", 
                 'deferral_payments', 'total_payments', 'loan_advances', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'long_term_incentive', 'restricted_stock', 'director_fees']
features_train, features_test, labels_train, labels_test = train_test_data(features_manual)

clf4 = DecisionTreeClassifier()
print '*** Decesiontree Algorithm with only payments features ***'
print test_algorithm(clf4,features_train,features_test), '\n'

## Parameter tuning
# Lets repeat 50%classifier with different classifier parameters to see 
# if we can achieve better result with any other parameter in algorithm
features_list = ["poi", "salary", "bonus", "fraction_from_poi_email", "fraction_to_poi_email",
                 'total_payments', 'total_stock_value', 'expenses', 'exercised_stock_options',
                 'shared_receipt_with_poi', 'restricted_stock']
features_train, features_test, labels_train, labels_test = train_test_data(features_list)

parameters = {'criterion':('gini', 'entropy')}
dtc = DecisionTreeClassifier()
clf5 = grid_search.GridSearchCV(dtc, parameters)
print 'Run Decesion Tree classifier with GridSearchCV'
print test_algorithm(clf5,features_train,features_test), '\n'
	
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
dump_classifier_and_data(clf0, my_dataset, features_li)