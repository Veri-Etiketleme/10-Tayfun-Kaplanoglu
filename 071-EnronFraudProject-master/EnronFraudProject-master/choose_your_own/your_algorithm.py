#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#print "REMINDER: YOU MUST CLOSE THE VISUALIZATION BEFORE CODE CAN CONTINUE"

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from time import time
model_type = None

#k-NN
'''
model_type = "k-NN"
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=20, weights='distance', algorithm='auto', leaf_size=30)
'''

#Random Forest
'''
model_type = "Random Forest"
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators = 10, max_depth=None, min_samples_split=10, min_samples_leaf=1, min_weight_fraction_leaf = 0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None)
'''

#AdaBoost

model_type = "AdaBoost"
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(n_estimators=10, learning_rate=0.1)



##############

t0 = time()
clf.fit(features_train, labels_train)
print "Training time for", model_type, ":", round(time()-t0, 3), "s"

t0 = time()
preds = clf.predict(features_test)
print "Testing time for", model_type, ":", round(time()-t0, 3), "s"


from sklearn.metrics import accuracy_score
acc = accuracy_score(labels_test, preds)
print "Accuracy =", acc

print "Don't forget to save the Trial filename!!"


try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
