#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

'''
We'll grab all of the features except the ones we know are more than 50%
missing values

Specifically, we're not going to pull in:
    1. loan_advances (97% missing)
    2. director_fees (88%)
    3. restricted_stock_deferred (87.7%)
    4. deferral_payments (73%)
    5. deferred_income (66%)
    6. long_term_incentive (54.9%)
    
Also, we won't pull in email_address as that doesn't give us any useful info
'''
features_list = ['poi', 'salary', 'total_payments', 'bonus', 
                 'total_stock_value', 
                 'expenses', 'exercised_stock_options', 'other',
                 'restricted_stock', 
                 'to_messages', 'from_poi_to_this_person', 
                 'from_messages', 'from_this_person_to_poi', 
                 'shared_receipt_with_poi'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Translating data_dict to a pandas DataFrame for ease of use
import pandas as pd
df = pd.DataFrame.from_dict(data_dict, orient = 'index')


### Task 2: Remove outliers
'''
Based upon the exploration we did in the Jupyter notebook included,
we know that we need to drop the records 'TOTAL' and 
'THE TRAVEL AGENCY IN THE PARK'.
'''
df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace = True)

#Here I'll also perform imputation to set all of the NaN values to 
#their feature-specific median
from sklearn.preprocessing import Imputer

imp = Imputer(missing_values='NaN', strategy='median')
#Need to get rid of emails first, can't impute those!
df.drop(columns = ['email_address'], inplace = True)
df.loc[:,:] = imp.fit_transform(df)

### Task 3: Create new feature(s)
	#I've included feature engineering as part of my Pipeline, please
		#see below for the Pipeline that includes it
	#Here I'll simply create the feature engineering class

from TopQuantile import TopQuantile


### Store to my_dataset for easy export below.
#I need to translate my data back into a dict for this step
data_dict = df.to_dict(orient = 'index')
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a variety of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#I'm only going to build the Pipeline using a k-Nearest Neigbhors
	#classifier built from GridSearchCV(), as I already explored
	#models in my Jupyter notebook, please see that for more info.

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

#It will be easier to work with these data not as a list,
#but as a numpy array
features = np.array(features)
labels = np.array(labels)


#Suppress the warnings coming from GridSearchCV to reduce output messages
import warnings
import sklearn.exceptions

warnings.filterwarnings("ignore",category=sklearn.exceptions.UndefinedMetricWarning)

#Shuffled and stratified cross-validation binning for this tuning exercise
cv_100 = StratifiedShuffleSplit(n_splits=100, test_size=0.1, random_state = 42)

#Imputation using the median of each feature
#imp = Imputer(missing_values='NaN', strategy='median')

#Feature Engineering with TopQuantile() to count the top quantile financial 
	#features
feats = [0,1,2,3,4,5,6,7]

topQ = TopQuantile(feature_list = feats)

#Feature Scaling via RobustScaler()
scaler = RobustScaler()

#Feature Selection via SelectPercentile(f_classif, percentile = 75)
selector = SelectPercentile(score_func = f_classif, percentile = 75)

#FeatureUnion to keep track of kNN and SVM model results
knn = KNeighborsClassifier()
knn_param_grid = {'kNN__n_neighbors': range(1,21,1), 
'kNN__weights': ['uniform', 'distance'], 'kNN__p': [1,2]}

#Hyperparameter tuning
from sklearn.model_selection import GridSearchCV

knn_pipe = Pipeline([('engineer',topQ), ('scale', scaler),
                    ('select', selector), ('kNN', knn)])

knn_gs = GridSearchCV(knn_pipe, knn_param_grid, scoring = ['precision', 'recall', 'f1'], 
                      cv = cv_100, refit = 'f1', return_train_score = False)

knn_gs.fit(features, labels)


results_df_knn = pd.DataFrame(knn_gs.cv_results_)
print "kNN:\n\n", results_df_knn.loc[knn_gs.best_index_]

clf = knn_gs.best_estimator_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)