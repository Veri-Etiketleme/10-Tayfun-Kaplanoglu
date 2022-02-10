# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

class detector:

	def __init__(self,save_path):
		prototxtPath = os.path.join("models","mobilenet", "deploy.prototxt")
		weightsPath = os.path.join("models","mobilenet","res10_300x300_ssd_iter_140000.caffemodel")
		self.net = cv2.dnn.readNet(prototxtPath, weightsPath)
		self.confidence = 0.9
		self.c = 0
		self.save_folder = save_path

	def run(self,image):
		# dimensions
		orig = image.copy()
		(h, w) = image.shape[:2]

		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
		self.net.setInput(blob)
		detections = self.net.forward()

		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with  the detection
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the confidence is
			# greater than the minimum confidence
			if confidence > self.confidence:
				# compute the (x, y)-coordinates of the bounding box for
				# the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# ensure the bounding boxes fall within the dimensions of
				# the frame
				(startX, startY) = (max(0, startX), max(0, startY))
				(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

				# extract the face ROI, convert it from BGR to RGB channel
				# ordering, resize it to 224x224, and preprocess it
				face = image[startY:endY, startX:endX]
				w = endY-startY
				h = endX-startX
				if (w > 0 and h > 0):

					#face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
					#face = cv2.resize(face, (224, 224))
					#face = img_to_array(face)

					cv2.imwrite(self.save_folder +str(self.c)+".png", face)
					self.c=self.c+1
					#cv2.waitKey(0)


				#face = preprocess_input(face)
				#face = np.expand_dims(face, axis=0)

class cascade:
	
	def __init__(self,save_path):
		self.dt = cv2.CascadeClassifier('models/haarcascade/haarcascade_frontalface_default.xml')
		self.c = 0
		self.save_folder = save_path

	def run(self, image):
		faces = self.dt.detectMultiScale(image, 1.1,5)
		if (len(faces)!=0):	
			for (x,y,w,h) in faces:
				roi = image[y:y+h,x:x+w]
				cv2.imwrite(self.save_folder+str(self.c)+".png", roi)
				self.c=self.c+1
				#cv2.waitKey(0)
				#cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)


def from_folder(folder_path = "dataset/mask", folder_save_path="test/"):
	try: 
		os.mkdir(folder_save_path) 
	except OSError as error: 
		print(error) 

	#folder_path = os.path.join("dataset", "with_mask")
	#folder_path = os.path.join("dataset", "without_mask")
	dt1 = detector(folder_save_path)
	dt2 = cascade(folder_save_path)

	names  = os.listdir(folder_path)
	for name in names:
		filename = os.path.join(folder_path, name)
		print(filename)
		image = cv2.imread(filename,1)
		dt1.run(image)

from_folder()