import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


if __name__ == "__main__":
	colmn_nms1=['LogAttackTime','HD','FluxD','spectral_bandwidthM','mfcc_1D','mfcc_3D','RMSM','spectral_bandwidthD',
		'mfcc_4M','mfcc_11D','ZCRD','spectral_centroidD','mfcc_8D','mfcc_6D','mfcc_7D','mfcc_4D','spectral_centroidM','mfcc_10M','mfcc_10D','Instrument']
	colmn_nms2=['spectral_centroidM','mfcc_2M','HC','ZCRM','mfcc_3M','HD','ZCRD','HS','mfcc_4M','mfcc_1M','mfcc_10M','FluxM','FluxD',
			'mfcc_5M','mfcc_7M','spectral_bandwidthM','spectral_centroidD','mfcc_6M','Instrument']
	colmn_nms3=['LogAttackTime','RMSM','mfcc_1D','mfcc_3D','mfcc_4D','mfcc_6D','mfcc_7D','mfcc_8D','mfcc_10D','mfcc_11D',
			'spectral_centroidM','spectral_centroidD','mfcc_1M','mfcc_2M','mfcc_3M','mfcc_4M','mfcc_5M','mfcc_6M','mfcc_7M','mfcc_10M',
			'ZCRM','ZCRD','HC','HD','HS','FluxM','FluxD','spectral_bandwidthM','spectral_bandwidthD','Instrument']

	instrdata_1 = pd.read_csv("dataset121.csv", names=colmn_nms1)
	instrdata_2 = pd.read_csv("dataset122.csv", names=colmn_nms2)
	instrdata_3 = pd.read_csv("dataset123.csv", names=colmn_nms3)
	instrdata_1.head()		
	X1 = instrdata_1.iloc[:, :9].values
	#print X1  
	y1 = instrdata_1.iloc[:,19].values
	#print y1
	instrdata_2.head()		
	X2 = instrdata_2.iloc[:, :9].values
	#print X2  
	y2 = instrdata_2.iloc[:,18].values
	#print y2
	instrdata_3.head()		
	X3 = instrdata_3.iloc[:, :-1].values
	#print X3  
	y3 = instrdata_3.iloc[:,29].values
	#print y3
	le = preprocessing.LabelEncoder()
	y1 = le.fit_transform(y1)

	y2 = le.fit_transform(y2)
	y3 = le.fit_transform(y3)
	#print y3
	#print le.inverse_transform(y3)

	X1_train, X1_test, y1_train,y1_test = train_test_split(X1, y1, test_size=0.44)
	X2_train, X2_test, y2_train,y2_test = train_test_split(X2, y2, test_size=0.44)
	X3_train, X3_test, y3_train,y3_test = train_test_split(X3, y3, test_size=0.44)	

	
	#from sklearn.model_selection import RepeatedKFold
	rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
	#cv=rkf
	# X is the feature set and y is the target
	'''for train_index, test_index in rkf.split(X):
		 print("Train:", train_index, "Validation:", val_index)
		 X_train, X_test = X[train_index], X[val_index]
		 y_train, y_test = y[train_index], y[val_index]'''

		 
				
	'''model=svm.SVC()
	#Hyper Parameters Set
	params = {'C': [6,7,8,9,10,11,12], 
			  'kernel': ['linear','rbf']}
	#Making models with hyper parameters sets
	model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
	#Learning
	model1.fit(train_X,train_y)
	#The best hyper parameters set
	print("Best Hyper Parameters:\n",model1.best_params_)
	#Prediction
	prediction=model1.predict(test_X)
	#importing the metrics module
	from sklearn import metrics
	#evaluation(Accuracy)
	print("Accuracy:",metrics.accuracy_score(prediction,test_y))
	#evaluation(Confusion Metrix)
	print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,test_y))'''

	
	#print np.size(Y1),np.size(Y2),np.size(Y3)
	#print le.inverse_transform(y1)

	scaler = StandardScaler()  
	#scaler.fit(X_train)
	X1_train = scaler.fit_transform(X1_train)  
	X1_test = scaler.fit_transform(X1_test)
	X2_train = scaler.fit_transform(X2_train)  
	X2_test = scaler.fit_transform(X2_test)
	X3_train = scaler.fit_transform(X3_train)  
	X3_test = scaler.fit_transform(X3_test)
	#print X1_train,X1_test

	mlp1 = MLPClassifier(hidden_layer_sizes=(7,7,7,7),momentum=0.2,max_iter=500)
	mlp2 = MLPClassifier(hidden_layer_sizes=(7,7,7,7),momentum=0.2,max_iter=500)
	mlp3 = MLPClassifier(hidden_layer_sizes=(15,15,15,15),momentum=0.2,max_iter=500)
	mlp1.fit(X1_train, y1_train)
	predictions1 = mlp1.predict(X1_test)
	#print "done training"

	print(confusion_matrix(y1_test,predictions1))  
	print(classification_report(y1_test,predictions1))

	mlp2.fit(X2_train, y2_train)
	predictions2 = mlp2.predict(X2_test)
	#print "done training"
	print(confusion_matrix(y2_test,predictions2))  
	print(classification_report(y2_test,predictions2))

	mlp3.fit(X3_train, y3_train)
	predictions3 = mlp3.predict(X3_test)
	#print "done training"
	print(confusion_matrix(y3_test,predictions3))  
	print(classification_report(y3_test,predictions3))
	'''colmn_nms=['spectral_centroidM','mfcc_2M','HC','ZCRM','mfcc_3M','HD','ZCRD','HS','mfcc_4M','mfcc_1M','mfcc_10M','FluxM','FluxD',
			'mfcc_5M','mfcc_7M','spectral_bandwidthM','spectral_centroidD','mfcc_6M']

	instrdata_2 = pd.read_csv("dataset102.csv", names=colmn_nms)
	instrdata_2.head()		
	X2 = instrdata_2.values
	X2=scaler.fit_transform(X2)
	#print X2
	pr= mlp2.predict(X2)
	print pr'''
	#print predictions3,predictions2,predictions1
	sv1 = SVC(C=10, kernel='rbf', degree=8)

	sv2 = SVC(C=10, kernel='rbf', degree=8)

	sv3 = SVC(C=10, kernel='rbf', degree=8)

	sv1.fit(X1_train, y1_train)
	predictions1 = sv1.predict(X1_test)
	#print "done training"

	print(confusion_matrix(y1_test,predictions1))  
	print(classification_report(y1_test,predictions1))

	sv2.fit(X2_train, y2_train)
	predictions2 = sv2.predict(X2_test)
	#print "done training"
	print(confusion_matrix(y2_test,predictions2))  
	print(classification_report(y2_test,predictions2))

	sv3.fit(X3_train, y3_train)
	predictions3 = sv3.predict(X3_test)
	#print "done training"
	print(confusion_matrix(y3_test,predictions3))  
	print(classification_report(y3_test,predictions3))
	'''

	

	k1 = KNeighborsClassifier(n_jobs=-1)
	k2 = KNeighborsClassifier(n_jobs=-1)
	k3 = KNeighborsClassifier(n_jobs=-1)
	cv=3

	#Hyper Parameters Set
	params = {'n_neighbors':[5,6,7,8,9,10],
	          'leaf_size':[1,2,3,5],
	          'weights':['uniform', 'distance'],
	          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
	          'n_jobs':[-1]}
	#Making models with hyper parameters sets
	print 1
	model1 = GridSearchCV(k1, param_grid=params, n_jobs=-1,cv=cv)
	model1.fit(X1_train,y1_train)
	#The best hyper parameters set
	print("Best Hyper Parameters:\n",model1.best_params_)
	prediction=model1.predict(X1_test)
	#importing the metrics module
	from sklearn import metrics
	#evaluation(Accuracy)
	print("Accuracy:",metrics.accuracy_score(prediction,y1_test))
	#evaluation(Confusion Metrix)
	print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y1_test))
	print 1
	model2 = GridSearchCV(k2, param_grid=params, n_jobs=-1,cv=cv)
	model2.fit(X2_train,y2_train)
	#The best hyper parameters set
	print("Best Hyper Parameters:\n",model2.best_params_)
	prediction=model1.predict(X2_test)
	#importing the metrics module
	#from sklearn import metrics
	#evaluation(Accuracy)
	print("Accuracy:",metrics.accuracy_score(prediction,y2_test))
	#evaluation(Confusion Metrix)
	print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y2_test))
	model3 = GridSearchCV(k3, param_grid=params, n_jobs=-1,cv=cv)
	model3.fit(X3_train,y3_train)
	#The best hyper parameters set
	print("Best Hyper Parameters:\n",model3.best_params_)
	prediction=model3.predict(X3_test)
	#importing the metrics module
	#from sklearn import metrics
	#evaluation(Accuracy)
	print("Accuracy:",metrics.accuracy_score(prediction,y3_test))
	#evaluation(Confusion Metrix)
	print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y3_test))
	print 1'''
	#Learning

	k=5

	k1 = KNeighborsClassifier(n_neighbors=k,n_jobs=-1,weights='distance',leaf_size=1,algorithm='ball_tree')
	k2 = KNeighborsClassifier(n_neighbors=k,n_jobs=-1,weights='distance',leaf_size=1,algorithm='ball_tree')
	k3 = KNeighborsClassifier(n_neighbors=k,n_jobs=-1,weights='distance',leaf_size=1,algorithm='ball_tree')



	k1.fit(X1_train, y1_train)
	predictions1 = k1.predict(X1_test)
	#print "done training"

	print(confusion_matrix(y1_test,predictions1))  
	print(classification_report(y1_test,predictions1))

	k2.fit(X2_train, y2_train)
	predictions2 = k2.predict(X2_test)
	#print "done training"
	print(confusion_matrix(y2_test,predictions2))  
	print(classification_report(y2_test,predictions2))

	k3.fit(X3_train, y3_train)
	predictions3 = k3.predict(X3_test)
	#print "done training"
	print(confusion_matrix(y3_test,predictions3))  
	print(classification_report(y3_test,predictions3))



	instrument_clasifier_pkl_filename = 'Instrument_classifier_mlp_1_4.pkl'
	# Open the file to save as pkl file
	models=[mlp1,mlp2,mlp3,sv1,sv2,sv3,k1,k2,k3]
	instrument_clasifier_pkl = open(instrument_clasifier_pkl_filename, 'wb')
	for model in models:
		pickle.dump(model,instrument_clasifier_pkl)

	
		# Close the pickle instances
	instrument_clasifier_pkl.close()