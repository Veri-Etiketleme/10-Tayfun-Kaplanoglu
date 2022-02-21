import sys
import os
sys.path.insert(0, os.path.abspath('..')) 
# from pprint import pprint as p
# p(sys.path)

# print os.environ['PYTHONPATH'].split(os.pathsep)
from utility.sklearnbasemodel import BaseModel
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from preprocess.preparedata import HoldoutSplitMethod




class LinearRegressionModel(BaseModel):
    def __init__(self):
        BaseModel.__init__(self)
#         self.usedFeatures = [101,102,103,4,5,6, 701,702,703,801,802,901,902]
        self.holdout_split = HoldoutSplitMethod.IMITTATE_TEST2_PLUS1
        self.save_final_model = False
        self.do_cross_val = False
        return
    def setClf(self):
#         self.clf = Ridge(alpha=0.0000001, tol=0.0000001)
        clf = LinearRegression()
        min_max_scaler = preprocessing.MinMaxScaler()
        self.clf = Pipeline([('scaler', min_max_scaler), ('estimator', clf)])
        return
    def after_train(self):
        print "self.clf.named_steps['estimator'].coef_:\n{}".format(self.clf.named_steps['estimator'].coef_)
        print "self.clf.named_steps['estimator'].intercept_:\n{}".format(self.clf.named_steps['estimator'].intercept_)
        return




if __name__ == "__main__":   
    obj= LinearRegressionModel()
    obj.run()