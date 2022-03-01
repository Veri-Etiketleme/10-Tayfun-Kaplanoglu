# -*- coding: utf-8 -*-

"""
Evalute the model
"""
from sklearn import metrics


def auc(y, pred_y):
    return metrics.roc_auc_score(y, pred_y)


def acc(y, pred_y):
    correct = 0
    wrong = 0

    for i in range(len(pred_y)):
        if pred_y[i] > 0.5 and y[i] == 1 or pred_y[i] < 0.5 and y[i] == 0:
            correct += 1
        else:
            wrong += 1

    print("True: %s ACC: %s" % (correct, float(correct)/float(correct+wrong)))



    
