import sys
import os
sys.path.insert(0, os.path.abspath('..'))

from implement.decisiontreemodel import DecisionTreeModel
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from evaluation.sklearnmape import mean_absolute_percentage_error_scoring
import logging
from utility.logger_tool import Logger
from datetime import datetime
from knnmodel import KNNModel
from utility.duration import Duration
from svmregressionmodel import SVMRegressionModel
from randomforestmodel import RandomForestModel
import numpy as np
from gradientboostingmodel import GrientBoostingModel
from xgboost_sklearnmodel import XGBoostSklearnModel


class TuneModel:
    """Grid search best paramers for sklearn based learning model
       Both exhaustive and random search are supported
       at the end ot he search, it will also save the model, predict on the test dataset.
    """
    def __init__(self):
        self.application_start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        logfile_name = r'logs/tunealgorithm_' +self.application_start_time + '.txt'
        _=Logger(filename=logfile_name,filemode='w',level=logging.DEBUG)
        self.durationtool = Duration()
        self.do_random_gridsearch = False
        self.n_iter_randomsearch = 2
        self.n_jobs = 1
        return
    def runGridSearch(self, model):
        logging.debug("run grid search on model: {}".format(model.__class__.__name__))
        logging.debug("cross validation strategy: {}".format(model.holdout_split))
        logging.debug("used features: {}".format(model.usedFeatures))
        logging.debug("tuned parameters: {}".format(model.getTunedParamterOptions()))
        
        features,labels,cv = model.getFeaturesLabel()
        # do grid search
        if self.do_random_gridsearch:
            estimator = RandomizedSearchCV(model.clf, model.getTunedParamterOptions(), cv=cv, n_jobs=self.n_jobs,
                       scoring=mean_absolute_percentage_error_scoring, verbose = 500, n_iter=self.n_iter_randomsearch)
        else:
            estimator = GridSearchCV(model.clf, model.getTunedParamterOptions(), cv=cv,n_jobs=-self.n_jobs, 
                                     fit_params=model.get_fit_params(),
                       scoring=mean_absolute_percentage_error_scoring, verbose = 500)
        estimator.fit(features, labels)
        model.clf = estimator.best_estimator_
        model.save_final_model = True
        model.save_model()
        
#         model.dispFeatureImportance()
        logging.debug('estimaator parameters: {}'.format(estimator.get_params))
        logging.debug('Best parameters: {}'.format(estimator.best_params_))
        logging.debug('Best Scores: {}'.format(-estimator.best_score_))
        logging.debug('Score grid: {}'.format(estimator.grid_scores_ ))
        for i in estimator.grid_scores_ :
            logging.debug('parameters: {}'.format(i.parameters ))
            logging.debug('mean_validation_score: {}'.format(np.absolute(i.mean_validation_score)))
            logging.debug('cv_validation_scores: {}'.format(np.absolute(i.cv_validation_scores) ))

        
        
        return
    def get_model(self, model_id):
        model_dict = {}
        model_dict[1] =DecisionTreeModel
        model_dict[2] =KNNModel
        model_dict[3] =SVMRegressionModel
        model_dict[4] = RandomForestModel
        model_dict[5] = GrientBoostingModel
        model_dict[6] = XGBoostSklearnModel
        return model_dict[model_id]()
    def run(self):
       
        model_id = 4

        model = self.get_model(model_id)
        model.application_start_time = self.application_start_time
        self.durationtool.start()
        self.runGridSearch(model)
        self.durationtool.end()
        return




if __name__ == "__main__":   
    obj= TuneModel()
    obj.run()