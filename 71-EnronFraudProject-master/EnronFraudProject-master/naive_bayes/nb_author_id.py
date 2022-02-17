#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#print features_train[:5]


#########################################################
### your code goes here ###

from sklearn.naive_bayes import GaussianNB

#Setup the classifier
clf = GaussianNB()

#Train the classifer
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"

#Predict an outcome, not necessary for the quiz at hand, but included for the sake of completeness
t0 = time()
predictions = clf.predict(features_test)
print "testing time:", round(time()-t0, 3), "s"

#Compute the accuracy of the trained classifer
print "Accuracy: {}".format([clf.score(features_test, labels_test)])


#########################################################


