# -*- coding: utf-8 -*-

"""
DNN for Competition
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from data_process import MatchData, tmp_load
from data_process import test_data_func, train_data_func
from models.dnn import DNN, SimDNN
import os
import numpy as np
import torch.optim as optimizer
import random
from evaluate import auc


CURRENT_COMP_VECTOR_SIZE = 2
TEAM_VECTOR_SIZE = 14
LEARNING_RATE = 0.0001


def save_model(net, name):
    path = 'model_params/'
    torch.save(net, path+name)


def load_model(name):
    path = 'model_params/'
    return torch.load(path+name)


def predict(model_name, home_vector, away_vector, home_state, away_state, opt):

    net = load_model(model_name)
    if opt.cuda == 1:
        home_current_state = Variable(torch.FloatTensor(home_state).cuda())
        away_current_state = Variable(torch.FloatTensor(away_state).cuda())

        away_vector = Variable(torch.FloatTensor(away_vector).cuda())
        home_vector = Variable(torch.FloatTensor(home_vector).cuda())
    else:
        home_current_state = Variable(torch.FloatTensor(home_state))
        away_current_state = Variable(torch.FloatTensor(away_state))

        away_vector = Variable(torch.FloatTensor(away_vector))
        home_vector = Variable(torch.FloatTensor(home_vector))

    home_current_state = home_current_state.unsqueeze(0)
    away_current_state = away_current_state.unsqueeze(0)
    home_vector = home_vector.unsqueeze(0)
    away_vector = away_vector.unsqueeze(0)

    prob = net(
        home_state=home_current_state,
        home_vector=home_vector,
        away_state=away_current_state,
        away_vector=away_vector
    )
    prob = prob.squeeze(0)
    return prob


def train_dnn_batch(epoches, team_data, opt):
    """
    train mini batch dnn here
    :return: 
    :rtype: 
    """

    batch_size = opt.batch_size
    LEARNING_RATE = 0.0001
    dnn = SimDNN(TEAM_VECTOR_SIZE)
    if opt.cuda == 1:
        dnn.cuda()
    #data_provider = MatchData(1000)
    dnn_optimizer = optimizer.RMSprop(dnn.parameters(), lr=LEARNING_RATE)
    prob_criterion = torch.nn.CrossEntropyLoss()
    score_criterion = torch.nn.MSELoss()
    train_data = train_data_func()
    if opt.dataset == "all":
        test_data = test_data_func()
        train_data += test_data
    print("Starting to train with DNN")
    for epoch in range(epoches):
        random.shuffle(train_data)
        if epoch == 35:
            LEARNING_RATE = LEARNING_RATE / 5.0
            for param_group in dnn_optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
        if epoch == 100:
            LEARNING_RATE = LEARNING_RATE / 10.0
            for param_group in dnn_optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
        for i in range(0, len(train_data)-batch_size, batch_size):
            batch_home_current_state = Variable(torch.zeros((batch_size, CURRENT_COMP_VECTOR_SIZE)))
            batch_away_current_state = Variable(torch.zeros((batch_size, CURRENT_COMP_VECTOR_SIZE)))

            batch_home_vector = Variable(torch.zeros((batch_size, TEAM_VECTOR_SIZE)))
            batch_away_vector = Variable(torch.zeros((batch_size, TEAM_VECTOR_SIZE)))

            batch_score = Variable(torch.zeros((batch_size, 2)))
            batch_result = []

            for p in range(batch_size):
                away_id = train_data[i+p][0]
                home_id = train_data[i+p][1]

                away_current_state = train_data[i+p][2:4]
                home_current_state = train_data[i+p][4:6]
                score = [train_data[i+p][7], train_data[i+p][6]]
                away_vector = team_data[away_id]
                home_vector = team_data[home_id]
                result = train_data[i+p][8]

                batch_home_current_state[p] = Variable(torch.FloatTensor(home_current_state))
                batch_away_current_state[p] = Variable(torch.FloatTensor(away_current_state))
                batch_away_vector[p] = Variable(torch.FloatTensor(away_vector))
                batch_home_vector[p] = Variable(torch.FloatTensor(home_vector))
                batch_score[p] = Variable(torch.FloatTensor(score))
                batch_result.append(result)
            batch_result = Variable(torch.LongTensor(batch_result))

            if opt.cuda == 1:
                batch_home_current_state = batch_home_current_state.cuda()
                batch_away_current_state = batch_home_current_state.cuda()
                batch_home_vector = batch_home_vector.cuda()
                batch_away_vector = batch_away_vector.cuda()
                batch_result = batch_result.cuda()
                batch_score = batch_score.cuda()

            output_prob = dnn.forward(
                home_vector=batch_home_vector,
                home_state=batch_home_current_state,
                away_vector=batch_away_vector,
                away_state=batch_away_current_state
            )
            loss = prob_criterion(output_prob, batch_result)
            #loss += 0.001*score_criterion(output_score, batch_score)

            dnn_optimizer.zero_grad()
            loss.backward()
            dnn_optimizer.step()

            if i % 100 == 0:
                print("Epoches: %s Sample: %s Loss: %s" % (epoch, i + 1, loss.data[0]))

        if opt.dataset == "train":
            save_model(dnn, 'train_dnn_%d.pkl' % epoch)
            opt.model_name = 'train_dnn_%d.pkl' % epoch
            test(team_data, opt)
        elif opt.dataset == "all":
            save_model(dnn, 'all_dnn_%d.pkl' % epoch)
        else:
            print("dataset error")


def train_dnn(epoches, team_data, opt):
    """
    train dnn here
    :return: 
    :rtype: 
    """
    LEARNING_RATE = 0.0001
    dnn = SimDNN(TEAM_VECTOR_SIZE)
    if opt.cuda == 1:
        dnn.cuda()
    #data_provider = MatchData(1000)
    dnn_optimizer = optimizer.RMSprop(dnn.parameters(), lr=LEARNING_RATE)
    prob_criterion = torch.nn.CrossEntropyLoss()
    score_criterion = torch.nn.MSELoss()
    train_data = train_data_func()
    if opt.dataset == "all":
        test_data = test_data_func()
        train_data += test_data

    print("Starting to train with DNN")
    for epoch in range(epoches):
        random.shuffle(train_data)
        if epoch == 35:
            LEARNING_RATE = LEARNING_RATE/5.0
            for param_group in dnn_optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
        if epoch == 100:
            LEARNING_RATE = LEARNING_RATE/10.0
            for param_group in dnn_optimizer.param_groups:
                param_group['lr'] = LEARNING_RATE
        for i in range(len(train_data)):
            #     Competition: [(Away, Home, Away_Ago_Win, Away_Ago_Lose, Home_Ago_Win, Home_Ago_Lose, Away_Score, Home_Score, Home_Win)]
            away_id = train_data[i][0]
            home_id = train_data[i][1]

            away_current_state = train_data[i][2:4]
            home_current_state = train_data[i][4:6]
            score = [train_data[i][7], train_data[i][6]]
            away_vector = team_data[away_id]
            home_vector = team_data[home_id]
            result = [train_data[i][8]]

            if opt.cuda == 1:
                home_current_state = Variable(torch.FloatTensor(home_current_state).cuda())
                away_current_state = Variable(torch.FloatTensor(away_current_state).cuda())

                away_vector = Variable(torch.FloatTensor(away_vector).cuda())
                home_vector = Variable(torch.FloatTensor(home_vector).cuda())
                prob = Variable(torch.LongTensor(result).cuda())
                score = Variable(torch.FloatTensor(score).cuda())
            else:    
                home_current_state = Variable(torch.FloatTensor(home_current_state))
                away_current_state = Variable(torch.FloatTensor(away_current_state))

                away_vector = Variable(torch.FloatTensor(away_vector))
                home_vector = Variable(torch.FloatTensor(home_vector))
                prob = Variable(torch.LongTensor(result))

                score = Variable(torch.FloatTensor(score))

            home_current_state = home_current_state.unsqueeze(0)
            away_current_state = away_current_state.unsqueeze(0)
            home_vector = home_vector.unsqueeze(0)
            away_vector = away_vector.unsqueeze(0)

            output_prob = dnn.forward(
                home_vector=home_vector,
                home_state=home_current_state,
                away_vector=away_vector,
                away_state=away_current_state
            )

            loss = prob_criterion(output_prob, prob)
            #loss += 0.001*score_criterion(output_score, score)

            # a, b = output_prob.data.cpu().numpy()[0][0], output_prob.cpu().data.numpy()[0][1]
            # if float(a)/float(b) > 9 or float(a)/float(b)< (1.0/9.0):
            #     for param_group in dnn_optimizer.param_groups:
            #         param_group['lr'] = LEARNING_RATE*10
                
            dnn_optimizer.zero_grad()
            loss.backward()
            dnn_optimizer.step()

            if i % 100 == 0:
                print("Epoches: %s Sample: %s Loss: %s" % (epoch, i+1, loss.data[0]))

        if opt.dataset == "train":
            save_model(dnn, 'train_dnn_%d.pkl' % epoch)
            opt.model_name = 'train_dnn_%d.pkl' % epoch
            test(team_data, opt)
        elif opt.dataset == "all":
            save_model(dnn, 'all_dnn_%d.pkl' % epoch)
        else:
            print("dataset error")


def test(team_data, opt):
    data_provider = MatchData(1000)
    data_provider.roll_data()

    #testing_data = data_provider.get_test_data()
    testing_data = test_data_func()

    log_file = open('testing.log', 'w+')

    correct = 0
    wrong = 0
    y_label = []
    y_pred = []
    for i in range(len(testing_data)):

        away_id = testing_data[i][0]
        home_id = testing_data[i][1]

        away_current_state = testing_data[i][2:4]
        home_current_state = testing_data[i][4:6]
        score = [testing_data[i][7], testing_data[i][6]]
        away_vector = team_data[away_id]
        home_vector = team_data[home_id]
        result = [testing_data[i][8]]

        prob = predict(
            opt.model_name,
            home_state=home_current_state,
            home_vector=home_vector,
            away_state=away_current_state,
            away_vector=away_vector,
            opt=opt
        )

        pred_win = np.argmax(prob.data.cpu().numpy())
        
        y_label.append(result)
        y_pred.append(prob.data.cpu().numpy()[1])


        if pred_win == result:
            correct += 1
            #line = 'Test: %s Correct! Confidence=%s' % (i, prob.data[pred_win])
        else:
            wrong += 1
            #line = 'Test: %s Wrong! Confidence=%s' % (i, prob.data.cpu().numpy().tolist())

        # print(line)
        # log_file.write(line+'\n')
    auc_score = auc(y_label, y_pred)
    print("TEST: auc score: %s" % auc_score)
    # log_file.close()

    print("Wrong: %s Correct: %s" % (wrong, correct))


def predict_result(team_data, opt):

    testing_data = tmp_load()
    output_file = open('output/predictPro.csv', 'rw+')
    output_file.write('主场赢得比赛的置信度\n')

    for i in range(len(testing_data)):

        away_id = testing_data[i][0]
        home_id = testing_data[i][1]

        away_current_state = testing_data[i][2:4]
        home_current_state = testing_data[i][4:6]
        away_vector = team_data[away_id]
        home_vector = team_data[home_id]

        prob = predict(
            opt.model_name,
            home_state=home_current_state,
            home_vector=home_vector,
            away_state=away_current_state,
            away_vector=away_vector,
            opt=opt
        )

        prob = prob.data[1]
        line = '%s' % prob
        output_file.write(line+'\n')

    output_file.close()






















