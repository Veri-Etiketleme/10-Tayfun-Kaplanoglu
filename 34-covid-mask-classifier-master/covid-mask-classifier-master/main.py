import os
import cv2
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, auc,precision_score,recall_score
import pickle
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from skl2onnx import convert_sklearn  
from skl2onnx.common.data_types import FloatTensorType
import joblib
from sklearn.metrics import classification_report
import onnxruntime as rt
from sklearn.pipeline import Pipeline

class classifier:
    
    def __init__(self):
        self.size = 32
        self.channel = 3
        self.folder_path = "dataset"

    def load_data(self):
        ## -----------  Read Data
        class_name =  ["mask", "no-mask"]

        num_classes = len(class_name)
        X = []
        Y = []
        for i in range(0,num_classes):
            folder_path = os.path.join(self.folder_path, class_name[i])
            for name in os.listdir(folder_path):
                path_read_im = os.path.join(folder_path, name)
                #print(path_read_im)
                im = None
                if self.channel==3: 
                    im = cv2.imread(path_read_im,1) # COLOR
                else:
                    im = cv2.imread(path_read_im,0) # GRAY

                im = cv2.resize(im, (self.size,self.size), interpolation = cv2.INTER_AREA)
                im = im.reshape(self.size*self.size*self.channel)

                X.append(im)
                Y.append(i)

        X = np.array(X, dtype="float32")
        Y = np.array(Y, dtype="float32")

        print(X.shape)
        print(Y.shape)
        return X,Y

    def compare_models(self,X,Y):
        # user variables to tune
        seed    = 5
        folds   = 5 # 10 = 10%, 5 = 20% for testing
        #5-fold cross validation. This means that 20% of the data is used for testing, this is usually pretty accurate.
        metric  ="roc_auc"

        # hold different regression models in a single dictionary
        models = {}
        models["GradientBoost"]        = GradientBoostingClassifier() #n_estimators=50)
        models["RandomForest"]         = RandomForestClassifier() #n_estimators=100)
        models["NaiveBayes"]           = GaussianNB()
        models["Logistic"]             = LogisticRegression()
        models["SVC"]                  = SVC(probability=True)

        # 10-fold cross validation for each model
        model_results = []
        model_names   = []
        for model_name in models:
            model   = models[model_name]
            k_fold  = KFold(n_splits=folds, random_state=seed,shuffle=True)
            results = cross_val_score(model, X, Y, cv=k_fold, scoring=metric)

            model_results.append(results)
            model_names.append(model_name)
            print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))

    def train(self,X,Y):
        X_train, X_test, Y_train, Y_test = 	(trainX, testX, trainY, testY) = train_test_split(X, Y, test_size=0.2, random_state=42)
	
        pipeline = Pipeline([
            ("sc", MinMaxScaler()), # 0-1 features
            #('polinomial', PolynomialFeatures(degree=3)),
            #("pca", PCA(n_components=0.98)),
            ("model", SVC(probability=True) )
        ])

        tuned_parameters = [
                    {'model__kernel': ['rbf'], 'model__gamma': [1e-3, 1e-4],  'model__C': [1, 10, 100, 1000]},
                    {'model__kernel': ['linear'], 'model__C': [1, 10, 100, 1000]},
                    {'model__kernel' : ['poly'], 'model__degree' : [2,3,4] ,'model__C': [1, 10, 100, 1000]}
                ]
        
        grid_search = RandomizedSearchCV(
           pipeline, tuned_parameters, scoring='f1_macro'
        )

        # Train
        grid_search.fit(X_train,Y_train)

        # Score
        score = grid_search.best_estimator_.score(X,Y)
        print("f1_weighted score : ", score)

        self.save_model_joblib("model.pkl", grid_search.best_estimator_)
        self.save_model_pickle("model.pkl", grid_search.best_estimator_)
        self.save_model_onnx("model.onnx", grid_search.best_estimator_)
        return grid_search.best_estimator_

    def report(self,X,Y):
        model = self.load_model_pickle("model.pkl")
        Y_pred =  model.predict(X)
        
        print("Report")
        print(classification_report(Y, Y_pred))

    def test_pickle(self,X):

        model = self.load_model_joblib("model.pkl")
        model = self.load_model_pickle("model.pkl")
        label = model.predict(X)
        res = model.predict_proba(X)
        
        for i in range(0, res.shape[0]):
            mask_perc = res[i][0]
            no_mask_perc = res[i][1]
            im = X[i,:].reshape(self.size,self.size,self.channel)
            #cv2.putText(im, "Mask: " + str(np.round(mask_perc,1))  ,(10,30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),1)
            cv2.imshow("im", np.uint8(im))
            print( "Mask: " + str(np.round(mask_perc,1)) )
            cv2.waitKey(0)
        
    def test_onnx(self,X):
        # Image has to be in range 0-255
        model_sess = self.load_model_onnx("model.onnx")
        input_name = model_sess.get_inputs()[0].name
        label_name = model_sess.get_outputs()[1].name  # 0 output_label, 1 output_proability
        res = model_sess.run([label_name], {input_name: X.astype(np.float32)})[0]
        for i in range(0, len(res)):
            mask_perc = res[i][0]
            no_mask_perc = res[i][1]
            im = X[i,:].reshape(self.size,self.size,self.channel)
            #cv2.putText(im, "Mask: " + str(np.round(mask_perc,1))  ,(10,30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),1)
            cv2.imshow("im", np.uint8(im))
            print( "Mask: " + str(np.round(mask_perc,1)) )
            cv2.waitKey(0)

    def test_real_time_onnx(self):
        # Image has to be in range 0-255
        model_sess = self.load_model_onnx("model.onnx")
        input_name = model_sess.get_inputs()[0].name
        label_name = model_sess.get_outputs()[1].name  # 0 output_label, 1 output_proability

        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            height, width, channels = frame.shape
            upper_left = (int(width / 3), int(height / 4))
            bottom_right = (int(width * 2.5 / 4), int(height * 3 / 4))
            cv2.rectangle(frame, upper_left, bottom_right, (255, 0, 0), 2)
            roi = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

            im = cv2.resize(roi, (self.size,self.size), interpolation = cv2.INTER_AREA)
            im = im.reshape(self.size*self.size*self.channel)
            im = im[np.newaxis,:]
            res = model_sess.run([label_name], {input_name: im.astype(np.float32)})[0]
          
            # Assume that there is only one face
            mask_perc = res[0][0]
            no_mask_perc = res[0][1]

            # Display the resulting frame
            cv2.putText(frame, "Mask: " + str(np.round(mask_perc,1))  ,(10,30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),1)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def test_real_time_pickle(self):
        # Image has to be in range 0-255
        model = self.load_model_joblib("model.pkl")
        model = self.load_model_pickle("model.pkl")
        label = model.predict(X)

        cap = cv2.VideoCapture(0)
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()

            height, width, channels = frame.shape
            upper_left = (int(width / 3), int(height / 4))
            bottom_right = (int(width * 2.5 / 4), int(height * 3 / 4))
            cv2.rectangle(frame, upper_left, bottom_right, (255, 0, 0), 2)
            roi = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]

            im = cv2.resize(roi, (self.size,self.size), interpolation = cv2.INTER_AREA)
            im = im.reshape(self.size*self.size*self.channel)
            im = im[np.newaxis,:]
            im = im.astype(np.float32)
            
            res = model.predict_proba(im)
            # Assume that there is only one face
            mask_perc = res[0][0]
            no_mask_perc = res[0][1]

            # Display the resulting frame
            cv2.putText(frame, "Mask: " + str(np.round(mask_perc,1))  ,(10,30), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0),1)
            cv2.imshow('frame',frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        cap.release()
        cv2.destroyAllWindows()

    def save_model_onnx(self,inp_name,inp_clf):
        # Convert into ONNX format
        initial_type = [('float_input', FloatTensorType([None, self.size*self.size*self.channel]))]
        onx = convert_sklearn(inp_clf, initial_types=initial_type)
        with open(inp_name, "wb") as f:
            f.write(onx.SerializeToString())

    def save_model_pickle(self,inp_name,inp_clf):
        #https://stackoverflow.com/questions/10592605/save-classifier-to-disk-in-scikit-learn
        with open(inp_name, 'wb') as f:
            pickle.dump(inp_clf, f) 

    def save_model_joblib(self,inp_name,inp_clf):
        joblib.dump(inp_clf, inp_name) 
    
    def load_model_joblib(self, inp_name):
        # Load a pipeline
        my_model_loaded = joblib.load(inp_name)
        return my_model_loaded

    def load_model_pickle(self,inp_name):
        with open(inp_name, 'rb') as f:
            model = pickle.load(f)
        return model

    def load_model_onnx(self,inp_name):
        model_onnx_sess =  rt.InferenceSession(inp_name)
        return model_onnx_sess


if __name__ == "__main__":
    
    cl = classifier()

    print("[LOAD DATA]")
    X,Y = cl.load_data()

    #print("[TRAIN]")
    #cl.train(X,Y)

    print("[REPORT]")
    cl.report(X,Y)

    print("[TEST]")
    #cl.test_pickle(X)
    #cl.test_onnx(X)
    cl.test_real_time_pickle()
    #cl.test_real_time_onnx()
    


    


    

    
