# -*- coding: utf-8 -*-

"""
Bayes for Competition
"""
import pickle
import numpy as np
from sklearn.naive_bayes import GaussianNB


class Bayes(object):

    def __init__(self):
        self.model = GaussianNB()

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict_proba(test_x)

    def save_model(self):
        with open('model_params/bayes.pkl', 'wb') as f:
            pickle.dump(self.model,f)

    def load_model(self):
        with open('model_params/bayes.pkl', 'wb') as f:
            self.model = pickle.load(f)
