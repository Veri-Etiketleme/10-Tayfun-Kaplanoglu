# -*- coding: utf-8 -*-

"""
Main
"""
import argparse
import train_dnn
import data_process as dp
import models.team as team
import train_svm as svm
import numpy as np


def get_team_representations(type):
    """
    Get Team Data
    :param type: representation type
    :type type: str
    :return: team data
    :rtype: dict()
    """
    team_data = dp.TeamData()
    team_data.process()
    team_raw_data = team_data.get_team()
    return team.team_representations(team_raw_data, type)


# DNN Model
def train_with_dnn(opt):
    print(opt.batch_size)
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.train_dnn_batch(opt.epoch, team_data, opt)


def test_with_dnn(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.test(team_data, opt)


def predict_dnn(opt):
    team_data = get_team_representations(opt.team_data_type)
    train_dnn.predict_result(team_data, opt)


### SVM Model
def train_svm(opt):
    team_data = get_team_representations(opt.team_data_type)
    svm.train(team_data, opt)


def predict_svm(opt):
    pass


def test_svm(opt):
    team_data = get_team_representations(opt.team_data_type)
    svm.test(team_data)

### XGBoost
def train_xgboost(opt):
    import train_xgboost as xgboost
    team_data = get_team_representations(opt.team_data_type)
    xgboost.train(team_data, opt)

def test_xgboost(opt):
    import train_xgboost as xgboost
    team_data = get_team_representations(opt.team_data_type)
    xgboost.test(team_data)

def predict_xgboost(opt):
    import train_xgboost as xgboost

    team_data = get_team_representations(opt.team_data_type)
    xgboost.predict(team_data)


def __test_team__():

    team_data = dp.TeamData()
    team_data.process()
    team_raw_data = team_data.get_team()

    team.average({1:team_raw_data[1], 2:team_raw_data[2]}, 10)

# if __name__ == '__main__':
#
#     __test_team__()




# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='basketball game prediction')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
#     parser.add_argument('--epoch', type=int, default=10,
#                     help='epoch number')               
#     parser.add_argument('--cuda', type=int, default=1,
#                     help='CUDA training')
#     parser.add_argument('--train', type=int, default=0,
#                     help='CUDA training')
#     parser.add_argument('--test', type=int, default=0,
#                     help='CUDA training')
#     parser.add_argument('--model', type=int, default=0,
#                     help='Choose model (dnn=0,svm=1,xgboost=2)')
#     parser.add_argument('--predict', type=int, default=0,
#                     help='predict result')                               
#     parser.add_argument('--model_param', type=str, default='epoch_30_params.pkl',
#                     help='model name')
#     parser.add_argument('--team_data_type', type=str, default='average',
#                     help='team data type')
#     args = parser.parse_args()

#     if args.train == 1:
#         if args.model == 0:
#             train_with_dnn(args)
#         elif args.model == 1:
#             train_svm(args)
#         elif args.model == 2:
#             train_xgboost(args)

#     if args.test == 1:
#         if args.model == 0:
#             test_with_dnn(args)
#         elif args.model == 1:
#             test_svm(args)
#         elif args.model == 2:
#             test_xgboost(args)

#     if args.predict == 1:
#         if args.model == 0:
#             predict_dnn(args)
#         elif args.model == 1:
#             predict_svm(args)
#         elif args.model == 2:
#             predict_xgboost(args)

