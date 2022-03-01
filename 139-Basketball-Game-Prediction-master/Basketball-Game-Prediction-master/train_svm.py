# -*- coding: utf-8 -*-

import models.svm as svm
from data_process import test_data_func, train_data_func
import numpy as np


def train(team_raw_data, opt):

    train_x = []
    train_y = []
    train_data = train_data_func()
    if opt.dataset == "all":
        test_data = test_data_func()
        train_data += test_data_func
    for x in train_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state

        train_x.append(input_vector)
        train_y.append(x[-1])

    __svm = svm.svm_model(train_x, train_y)
    
    if opt.dataset == "train":
        svm.save_model(__svm, 'train_svm_%d.pkl.pkl')
    elif opt.dataset == "all":
        svm.save_model(__svm, 'all_svm_%d.pkl.pkl')
    else:
        print("choose dataset error")


def test(team_raw_data):
    __test_data = test_data_func()
    svm_model = svm.load_model('svm_model_params.pkl')
    log_file = open('log/svm.log', 'w+')
    correct = 0
    wrong = 0
    for x in __test_data:
        home = x[1]
        away = x[0]
        home_vector = team_raw_data[home]
        away_vector = team_raw_data[away]

        away_state = x[2:4]
        home_state = x[4:6]

        input_vector = home_vector.tolist() + home_state + away_vector.tolist() + away_state

        pred = svm.predict(svm_model, input_vector)
        print(pred)
        pred_id = 1
        if pred < 0.5:
            pred_id = 0
        line = 'SVM: Pred:%s Real: %s Confidence=%s' % (pred_id, x[-1], pred)
        if pred_id == x[-1]:
            correct += 1
        else:
            wrong += 1
        print(line)
        log_file.write(line+'\n')

    print("Correct: %s Wrong: %s" % (correct, wrong))
    print("Acc=%s" % (float(correct)/float(correct+wrong)))
    log_file.close()









