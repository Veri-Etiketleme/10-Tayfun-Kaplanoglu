import argparse
import os

import cv2
import numpy as np

from joblib import dump, load
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

def prepareInput(imgPath):
    features, labels = [], []

    for i in range(0, 10):
        curPath = os.path.join(imgPath, str(i))
        files = os.listdir(curPath)
        for f in files:
            img = cv2.imread(os.path.join(curPath, f))
            features.append(img[:, :, 0].flatten())
            labels.append(i)
            print(os.path.join(curPath, f))

    return features, labels

def trainModel(X, y, n):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, verbose=1)
    clf.fit(X_train, y_train) 
    print(clf.score(X_test, y_test))

    clfall = RandomForestClassifier(n_estimators=200)
    clfall.fit(X, y)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    dump(clfall, n)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--imgPath", 
            required = True, 
            help = "Path to images")
    ap.add_argument("-n", "--outputName", 
            required = False, 
            default = 'classifier.joblib',
            help = "Output filename")
    args = vars(ap.parse_args())
    trainModel(*prepareInput(args['imgPath']), args['outputName'])
