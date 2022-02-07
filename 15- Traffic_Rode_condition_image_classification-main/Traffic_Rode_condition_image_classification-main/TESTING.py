# OWN DATA SET IMAGE CLASSIFICATION USING LENET-ARCHITECTURE MODEL:-


#Importing keras libraries and packages
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import cv2
import pickle
import imutils

# Load Model:-
model = load_model("TRAINING_EXPERIENCE.h5")
mlb = pickle.loads(open("mlb.pickle", "rb").read())

# Read an Input image:-
image = cv2.imread('INPUT/2.jpg')
output = imutils.resize(image,width=400)
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)
proba = model.predict(image)[0]
print(proba)
idxs = np.argsort(proba)[::-1][:2]
print(idxs)

for (i, j) in enumerate(idxs):
	label = "{}: {:.2f}%".format(mlb.classes_[j], proba[j] * 100)
	print(mlb.classes_[j])
	print(label)
	cv2.putText(output, label, (10, (i * 30) + 25), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow('Output_image',output)
