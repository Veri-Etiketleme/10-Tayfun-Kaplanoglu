#!/usr/bin/env python
# coding: utf-8

# # Comparision of ML classification models to predict the presence of heart disease in the patient

# <b>Data:</b>
# This database contains 76 attributes, but all published experiments refer to using a subset of 14 of them. <br>
# Attribute Information:<br>
# age<br>
# sex<br>
# chest pain type (4 values)<br>
# resting blood pressure</br>
# serum cholestoral in mg/dl<br>
# fasting blood sugar > 120 mg/dl<br>
# resting electrocardiographic results (values 0,1,2)<br>
# maximum heart rate achieved<br>
# exercise induced angina<br>
# oldpeak = ST depression induced by exercise relative to rest<br>
# the slope of the peak exercise ST segment<br>
# number of major vessels (0-3) colored by flourosopy<br>
# thal: 3 = normal; 6 = fixed defect; 7 = reversable defect<br>
# The "target" field refers to the presence of heart disease in the patient. It is integer 0 or 1(presence).</br>

# <h2> 1. Importing modules </h2>

# In[1]:


import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, mean_squared_error, precision_recall_fscore_support 
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
from hyperopt import hp, STATUS_OK, tpe, Trials, fmin

import xgboost as xgb


# <h2> 2. Reading data </h2>

# In[2]:


data = pd.read_csv(r"heart_disease_dataset.csv", sep = ',')


# In[3]:


data.head()


# <h2> 3. Display correlation matrix </h2>

# In[4]:


corr_matrix = data.corr()
corr_matrix


# <h2> 4. Change type of categorical attributes </h2>

# In[5]:


cat_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope','target']
for cat_col in cat_cols:
    data[cat_col] = data[cat_col].astype('category')
data.dtypes


# In[6]:


data.dtypes


# <h2> 5. Dealing with misssing values </h2>
# Some values were dropped manually to make data more non-trivial

# In[7]:


data.isnull().sum()


# In[8]:


cols_with_missing = [col for col in data.columns
                     if data[col].isnull().any()]
cols_with_missing


# In[9]:


cols = data.columns


# Below I used SimpleImputer to replace missing data. I needed to use two different strategies - for cathegorical attributes I used 'most_frequent' strategy and for float values I used 'mean' strategy. 

# In[10]:


for col in cols:
    if col in cat_cols:
        imputer = SimpleImputer(strategy = 'most_frequent')
        data[col] = pd.DataFrame(imputer.fit_transform(np.array(data[col]).reshape(-1, 1)))
    else:
        imputer = SimpleImputer(strategy = 'mean')
        data[col] = pd.DataFrame(imputer.fit_transform(np.array(data[col]).reshape(-1, 1)))
        scaler = StandardScaler()
        data[col] = pd.DataFrame(scaler.fit_transform(np.array(data[col]).reshape(-1, 1)))
        

    
data.columns = cols

data.head()


# Here I checked if all missing vales were properly replaced.

# In[11]:


data.isnull().sum()


# In[12]:


cat_cols = ['sex', 'chest pain type', 'fasting blood sugar', 'resting ecg', 'exercise angina', 'ST slope','target']
for cat_col in cat_cols:
    data[cat_col] = data[cat_col].astype('category')
data.dtypes


# <h2> 6. Encoding categorical attributes  </h2>

# In[13]:


data = pd.get_dummies(data, drop_first=True)


# <h2> 7. Checking class distribution  </h2>

# In[14]:


class_distribution = data["target_1"].value_counts()
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1000,100))
plt.ylabel("Amount of samples")
plt.xlabel("Classes")
sns.barplot(x= ['Positive', 'Negative'], y = class_distribution)
plt.show()


# <h2> 8. Split data into train and test set. Ratio 8:2 </h2>

# In[15]:


X_train, X_test, y_train, y_test = train_test_split(data.drop('target_1', 1), data['target_1'], test_size = 0.2, random_state=42)


# In[17]:


def calculate_results(model, X_test=X_test, y_test=y_test):
    start = time.time()
    y_predict = model.predict(X_test)
    y_pred_quant = model.predict_proba(X_test)[:, 1]
    y_pred_bin = model.predict(X_test)
    end = time.time()
    eltime = end - start
    
    mse = mean_squared_error(y_test, y_pred_bin)
    cm = confusion_matrix(y_test, y_pred_bin)
    print(cm)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.rcParams['font.size'] = 12
    plt.title('ROC curve for Heart Disease classifier')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.grid(True)
    AUC =  auc(fpr, tpr)
    print('AUC', AUC)
    print(classification_report(y_test, y_pred_bin))
    
    return {'eltime' : eltime, 'fpr' : fpr, 'tpr' : tpr, 'auc' : AUC, 'confusion_matrix' : cm, 'mse' : mse}


# <h2> 9. Random Forest Classifier  </h2>

# In[18]:


param_grid = [
    {'n_estimators': [2, 4, 5, 8, 10],'max_features': [3,  8, 9, 12], 'criterion': ['gini', 'entropy']}
]

forest_reg = RandomForestClassifier()
crf = GridSearchCV(forest_reg, param_grid, cv=5, scoring = 'f1')
crf.fit(X_train, y_train)
print(crf.best_params_)
print(crf.best_estimator_)
cvres = crf.cv_results_
crf_best = crf.best_estimator_


# In[23]:


crf_results = calculate_results(crf_best, X_test=X_test, y_test=y_test)


# <h2> 10. Logistic Regression Classifier  </h2>

# In[20]:


param_grid_clgr = [
    {"C":np.logspace(-3,2,6)}
]
clgr_reg = LogisticRegression()
clgr = GridSearchCV(clgr_reg, param_grid_clgr, cv=5, scoring = 'neg_mean_squared_error')
clgr.fit(X_train, y_train)
print(clgr.best_params_)
print(clgr.best_estimator_)
clgr_res = clgr.cv_results_
for mean_score, params in zip(clgr_res["mean_test_score"], clgr_res["params"]):
    print(np.sqrt(-mean_score), params)
clgr_best = clgr.best_estimator_


# In[22]:


clgr_results = calculate_results(clgr_best, X_test=X_test, y_test=y_test)


# <h2> 11. KNNeighbors Classifier  </h2>

# In[25]:


param_grid_knn = [
    {"n_neighbors": [3, 5, 7, 9, 11, 13, 15, 20, 25, 30, 35, 40, 50], 'weights': ['uniform', 'distance'], 'metric':['euclidean', 'manhattan']}
]
knn = KNeighborsClassifier()
knn = GridSearchCV(knn, param_grid_knn, cv=5, scoring = 'f1')
knn.fit(X_train, y_train)
print(knn.best_params_)
print(knn.best_estimator_)
knn_res = knn.cv_results_
knn_best = knn.best_estimator_


# In[26]:


cknn_results = calculate_results(knn_best, X_test=X_test, y_test=y_test)


# <h2> 12. Decision Tree Classifier  </h2>

# In[27]:


param_grid_cdt = [
    {'max_leaf_nodes': list(range(2, 10)), 'min_samples_split': [2, 3, 4, 5, 6]}
]
cdt_reg = DecisionTreeClassifier()
cdt = GridSearchCV(cdt_reg, param_grid_cdt, cv=5, scoring = 'f1')
cdt.fit(X_train, y_train)
print(cdt.best_params_)
print(cdt.best_estimator_)
cdt_res = cdt.cv_results_
for mean_score, params in zip(cdt_res["mean_test_score"], cdt_res["params"]):
    print(np.sqrt(-mean_score), params)
cdt_best = cdt.best_estimator_


# In[29]:


cdt_results = calculate_results(cdt_best, X_test=X_test, y_test=y_test)


# <h2> 13. Support Vector Machine Classifier  </h2>

# In[31]:


param_grid_csvm = [
    {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001]}
]
csvm_reg = SVC(probability = True)
csvm = GridSearchCV(csvm_reg, param_grid_csvm, cv=5, scoring = 'neg_mean_squared_error')
csvm.fit(X_train, y_train)
print(csvm.best_params_)
print(csvm.best_estimator_)
csvm_res = csvm.cv_results_
for mean_score, params in zip(csvm_res["mean_test_score"], csvm_res["params"]):
    print(np.sqrt(-mean_score), params)
csvm_best = csvm.best_estimator_


# In[32]:


csvm_results = calculate_results(csvm_best, X_test=X_test, y_test=y_test)


# <h2> 14. Naive Bayes Classifier - Gaussian  </h2>

# In[33]:


cnb = GaussianNB()
cnb.fit(X_train, y_train)


# In[35]:


cnb_results = calculate_results(cnb, X_test=X_test, y_test=y_test)


# <h2> 15. Neural Network with Hyperopt </h2>

# In[36]:


def get_model(params):
    return Sequential([
    Dense(int(params['filters']), activation='relu', input_shape = (16,),  kernel_regularizer=regularizers.l2(0.001)),
    Dropout(params['dropout_one']),
    Dense(int(params['filters_two']), activation='relu',  kernel_regularizer=regularizers.l2(0.001)),
    Dropout(params['dropout_two']),
    Dense(int(params['filters_three']), activation='relu',  kernel_regularizer=regularizers.l2(0.001)),
    Dropout(params['dropout_three']),
    Dense(2, activation='softmax')
    ])


# In[37]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.2, random_state=42)
def fun_obj(params):
    model = get_model(params)
    model.summary()
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = params.get('optimizer','RMSprop'), metrics = ['acc'])
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5)
    history = model.fit(X_train, y_train, batch_size=int(params.get('batch_size', 16)), validation_data = (X_val, y_val),
    steps_per_epoch=len(X_train) / int(params.get('batch_size', 16)), epochs = int(params.get('epochs', 25)), callbacks=[callback])    
    score = model.evaluate(X_test, y_test)
    y_pred = model.predict_classes(X_test)
    y_pred_proba = model.predict_proba(X_test)
    fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test, y_pred_proba[:, 1])
    AUC_cnn = auc(fpr_cnn, tpr_cnn)
    accuracy = score[1]
    results = {accuracy : params}
    print(classification_report(y_test, y_pred), 'AUC', AUC_cnn)
    cm_cnn = confusion_matrix(y_test, y_pred)
    print(cm_cnn)
    return {'loss': -AUC_cnn, 'status': STATUS_OK, 'model': model, 'results' : results}


# In[38]:


space = {
    'batch_size': hp.quniform('batch_size', 16, 64, 4),
    'epochs': hp.quniform('epochs', 4, 200, 4),
    'filters': hp.quniform('filters', 40, 1024, 20),
    'dropout_one': hp.quniform('dropout_one', 0.1, 0.7, 0.1),
    'filters_two': hp.quniform('filters_two', 40, 1024, 20),
    'dropout_two': hp.quniform('dropout_two', 0.1, 0.7, 0.05),
    'filters_three': hp.quniform('filters_three', 40, 1024, 20),
    'dropout_three': hp.quniform('dropout_three', 0.1, 0.7, 0.1),
    'optimizer': hp.choice('optimizer', ['Adam', 'RMSprop', 'SGD'])
}
tpe_trials = Trials()
best = fmin(
    fun_obj,
    space,
    tpe.suggest,
    max_evals = 15,
    trials = tpe_trials
)
print(best)


# In[39]:


model = Sequential([
    Dense(int(best['filters']), activation='relu', input_shape = (16,),  kernel_regularizer=regularizers.l2(0.001)),
    Dropout(best['dropout_one']),
    Dense(int(best['filters_two']), activation='relu',  kernel_regularizer=regularizers.l2(0.001)),
    Dropout(best['dropout_two']),
    Dense(int(best['filters_three']), activation='relu',  kernel_regularizer=regularizers.l2(0.001)),
    Dropout(best['dropout_three']),
    Dense(2, activation='softmax')
])
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='RMSprop', metrics=['acc'])
callback = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=10)
history = model.fit(X_train, y_train, batch_size=int(best['batch_size']), validation_data=(X_val, y_val),
                    steps_per_epoch=len(X_train) / int(best['batch_size']), epochs=400, callbacks=[callback])    
score = model.evaluate(X_test, y_test)
start = time.time()
y_pred = model.predict_classes(X_test)
y_pred_proba = model.predict_proba(X_test)
end = time.time()
eltime_cnn = end - start
fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_test, y_pred_proba[:, 1])
AUC_cnn = auc(fpr_cnn, tpr_cnn)
accuracy = score[1]
results = {accuracy : params}
print(classification_report(y_test, y_pred), 'AUC', AUC_cnn)
cm_cnn = confusion_matrix(y_test, y_pred)
print(cm_cnn)


[pre_cnn, rec_cnn, fsc_cnn, supp_cnn] = precision_recall_fscore_support(y_test, y_pred)
mse_cnn = mean_squared_error(y_test, y_pred)


# In[40]:


acc = history.history['acc']
loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(18, 8))
plt.subplot(1, 2, 1)
plt.plot(range(len(acc)), acc, label='Training Accuracy')
plt.plot(range(len(val_acc)), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(len(loss)), loss, label='Training Loss')
plt.plot(range(len(val_loss)), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training Loss')
plt.show()


# In[44]:


cnn_results = {'eltime' : eltime_cnn, 'fpr' : fpr_cnn, 'tpr' : tpr_cnn, 'auc' : AUC_cnn, 'confusion_matrix' : cm_cnn, 'mse' : mse_cnn}


# <h2> 16. XGradientBoost Classifier  </h2>

# In[45]:


param_grid_cxgb = [{
     "eta"    : [0.01, 0.05, 0.15] ,
     "max_depth"        : [ 7, 9, 11],
     "min_child_weight" : [ 1, 3, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.3],
     "colsample_bytree" : [ 0.3, 0.5],
     "min_samples_split": np.linspace(0.1, 0.5, 5),
     "min_samples_leaf": np.linspace(0.1, 0.5, 5),
     }]
cxgb_reg = xgb.XGBClassifier()
cxgb = GridSearchCV(cxgb_reg, param_grid_cxgb, cv=5, scoring='f1')
cxgb.fit(X_train, y_train)
print(cxgb.best_params_)
print(cxgb.best_estimator_)
xgb_res = cxgb.cv_results_
xgb_best = cxgb.best_estimator_


# In[46]:


xgb_results = calculate_results(xgb_best, X_test=X_test, y_test=y_test)


# <h2> 17. Comparision AUCs of Classifiers  </h2>

# In[50]:


colors = ["yellow", "green", "orange", "magenta","cyan","black", "blue", "red"]
legend = ["Random Forest\nClassifier", "Linear\nRegression", "K-Nearest\nNeightbours", "Decision Tree\nClassifier", "Support Vector\nMachine",
         "Naive Bayes\nClassifier", "Neural Network", "XGradientBoost\nClassifier"]
AUCs = [crf_results['auc'], clgr_results['auc'], cknn_results['auc'], cdt_results['auc'],  csvm_results['auc'],  cnb_results['auc'], cnn_results['auc'], xgb_results['auc']]
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1,0.1))
plt.ylabel("AUC")
plt.xlabel("Classifiers")
sns.barplot(x=legend, y=AUCs, palette=colors)
plt.show()


# <h2> 18. Comparision Mean Squared Errors of Classifiers  </h2>

# In[51]:


mses = [crf_results['mse'], clgr_results['mse'], cknn_results['mse'], cdt_results['mse'],  csvm_results['mse'],  cnb_results['mse'], cnn_results['mse'], xgb_results['mse']]
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,1,0.1))
plt.ylabel("MSE")
plt.xlabel("Classifiers")
sns.barplot(x=legend, y=mses, palette=colors)
plt.show()


# <h2> 19. Comparision times of Classifiers predictions  </h2>
# 

# In[56]:


eltimes = [crf_results['eltime'], clgr_results['eltime'], cknn_results['eltime'], cdt_results['eltime'],  csvm_results['eltime'],  cnb_results['eltime'], cnn_results['eltime'], xgb_results['eltime']]
sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,max(eltimes),0.01))
plt.ylabel("Prediction Time")
plt.xlabel("Classifiers")
sns.barplot(x= legend, y= eltimes, palette=colors)
plt.show()


# <h2> 20. Comparision Confusion Matricss of Classifiers  </h2>

# In[59]:


plt.figure(figsize=(240,120))

plt.suptitle("Confusion Matrixes",fontsize=600)
plt.subplots_adjust(wspace = 1, hspace= 1)

plt.subplot(4,2,1)
plt.title(legend[0],fontsize=200)
sns.heatmap(crf_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,2)
plt.title(legend[1],fontsize=200)
sns.heatmap(clgr_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d",cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,3)
plt.title(legend[2],fontsize=200)
sns.heatmap(cknn_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,4)
plt.title(legend[3],fontsize=200)
sns.heatmap(cdt_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,5)
plt.title(legend[4],fontsize=200)
sns.heatmap(csvm_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,6)
plt.title(legend[5],fontsize=200)
sns.heatmap(cnb_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,7)
plt.title(legend[6],fontsize=200)
sns.heatmap(cnn_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})

plt.subplot(4,2,8)
plt.title(legend[7],fontsize=200)
sns.heatmap(xgb_results['confusion_matrix'], annot=True, cmap="Blues", fmt="d", cbar=False, annot_kws={"size": 120})
plt.show()


# <h2> 21. Summary </h2>
# 

# ### In this project I compared eight different Machine Learning Classifiers. For each of models I used GridSearchCV to find the best hyperparameters. For CNN model I used HyperOpt module to optimize hyperparameters. The worst models are definitely Decision Tree Classifier and Gaussian Naive Bayes Classifier. The higher AUC has kNN, SVC, RandomForest Classifier and Neural Networks. To final classfication model I would choose kNN or RandomForestClassifier due to high recall value which is important during preparing models to predict diseases. XGBoost also obtained high recall value and can be considered in the future modeling.
# 

# In[ ]:




