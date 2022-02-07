import matplotlib
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ARCHITECTURE.lenet import LeNet
from imutils import paths
import numpy as np
import random
import pickle
import cv2
import os

# Parameters:-
EPOCHS = 75
INIT_LR = 1e-3
BS = 32
IMAGE_DIMS = (96, 96, 3)

# Input image:-
imagePaths = sorted(list(paths.list_images("Dataset")))
print(imagePaths)

random.seed(42)
random.shuffle(imagePaths)

# Create an list:-
data=[]
labels=[]

# loop over the input images
for imagePath in imagePaths:
	image = cv2.imread(imagePath)
	image = cv2.resize(image,(IMAGE_DIMS[1],IMAGE_DIMS[0]))
	image = img_to_array(image)
	data.append(image)

	l = label = imagePath.split(os.path.sep)[-2].split("_")
	labels.append(l)

# Convert into numpy both input & lables:-
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# binarizer implementation
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

#  loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
	print("{}. {}".format(i + 1, label))

# Split Training & Testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.2, random_state=42)

# Model:-
model = LeNet.build(numChannels=IMAGE_DIMS[2],imgRows=IMAGE_DIMS[0], imgCols=IMAGE_DIMS[1],numClasses=2,activation="relu",weightsPath=None)

#compile 
model.compile(loss = keras.losses.categorical_crossentropy,optimizer = 'SGD',metrics = ['accuracy'])

# fitting the model 
hist = model.fit(x=trainX,y=trainY,epochs = 50,batch_size = 128,validation_data =(testX,testY),verbose = 1)

# evaluate the model
test_score = model.evaluate(testX,testY)
print("Test loss {:.5f},accuracy {:.3f}".format(test_score[0],test_score[1]*100))

# Save the model
model.save('TRAINING_EXPERIENCE.h5')
f = open("MLB.PICKLE", "wb")
f.write(pickle.dumps(mlb))
f.close()


