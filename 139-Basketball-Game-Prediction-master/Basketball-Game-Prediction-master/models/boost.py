# -*- coding: utf-8 -*-

import xgboost as xgb


class Boost(object):

    def __init__(self):
        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'early_stopping_rounds': 100,
            'scale_pos_weight': 1,
            'eval_metric': 'auc',
            'gamma': 0.2,
            'max_depth': 15,
            'lambda ': 100,
            'subsample': 1,
            'colsample_bytree': 0.5,
            'min_child_weight': 5,
            'eta': 0.05,
            'seed': 2321531,
            'nthread': 2,
            'max_delta_step': 1
        }

    def train(self, train_x, train_y):
        d_train = xgb.DMatrix(train_x, label=train_y)
        self.model = xgb.train(self.params, d_train, num_boost_round=5000)

    def predict(self, x):
        x = xgb.DMatrix(x)
        pred_y = self.model.predict(x)
        return pred_y

    def save_model(self):
        self.model.save_model('model_params/xgboost_model_params.model')

    def load_model(self):
        self.model = xgb.Booster({'nthread':1})
        self.model.load_model('model_params/xgboost_model_params.model')
        pass

