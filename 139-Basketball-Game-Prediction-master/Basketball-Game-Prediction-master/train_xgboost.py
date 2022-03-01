# -*- coding: utf-8 -*-

from data_process import test_data_func, train_data_func, tmp_load
import numpy as np
import models.boost as boost
from evaluate import auc, acc


def train(team_raw_data, opt):

    train_x = []
    train_y = []
    train_data = train_data_func()
    if opt.dataset == "all":
        test_data = test_data_func()
        train_data += test_data
    __boost = boost.Boost()
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]
        input_vector = (home_vector-away_vector).tolist() + home_state + away_state

        train_x.append(input_vector)
        train_y.append(x[-1])

    __boost.train(train_x, train_y)
    __boost.save_model()


def test(team_raw_data):
    test_x = []
    y = []
    train_data = test_data_func()
    __boost = boost.Boost()
    __boost.load_model()
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state
        y.append(x[-1])
        test_x.append(input_vector)
    pred_y = __boost.predict(test_x)
    auc_ = auc(y, pred_y)
    acc(y, pred_y)
    print("AUC:%s" % auc_)

    with open('log/xgboost_test.log', 'w+') as f:
        for i in range(len(pred_y)):
            f.write('%s,%s' % (y[i], pred_y[i]))
            f.write('\n')


def predict(team_raw_data):

    testing_data = tmp_load()
    output_file = open('output/predictPro.csv', 'w+')
    output_file.write('主场赢得比赛的置信度\n')

    __boost = boost.Boost()
    __boost.load_model()
    testing_input = []
    for x in testing_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state
        testing_input.append(input_vector)

    pred = __boost.predict(testing_input)
    for prob in pred:
        line = '%s\n' % prob
        output_file.write(line)

    output_file.close()








