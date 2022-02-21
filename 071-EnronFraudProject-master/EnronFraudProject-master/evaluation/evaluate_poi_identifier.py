#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import numpy as np

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.model_selection import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, labels, test_size=0.30, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(feature_train, target_train)
preds = clf.predict(feature_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(target_test, preds)

print "Accuracy =", acc
print "# of test set POIs =", np.sum(target_test)
print "# of people in test set =", len(target_test)

from sklearn.metrics import precision_score, recall_score
score_p = precision_score(target_test, preds, pos_label =1, average = 'binary')
score_r = recall_score(target_test, preds, pos_label =1, average = 'binary')

print "Recall =", score_r
print "Precision =", score_p
