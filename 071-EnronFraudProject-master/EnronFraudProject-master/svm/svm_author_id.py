#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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




#########################################################
### your code goes here ###

from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC(C = 10000.0, kernel = 'rbf')

#Cut down training data set by 99% to see how it affects speed and accuracy
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train,labels_train)
print "Training time:", round(time()-t0, 3), "s"

t0 = time()
preds = clf.predict(features_test)
print "Testing time:", round(time()-t0, 3), "s"

print "Accuracy: ", accuracy_score(labels_test, preds)

print "Data classes: Sara = 0, Chris = 1"
#print "Predicted data class for element 10: ", preds[10]
#print "Predicted data class for element 26: ", preds[26]
#print "Predicted data class for element 50: ", preds[50]

#Since predictions for Sara result in 0, summing up will be the fastest way to identify Chris predictions
print "Total number of Chris (1) predictions = ", preds.sum()

#########################################################


