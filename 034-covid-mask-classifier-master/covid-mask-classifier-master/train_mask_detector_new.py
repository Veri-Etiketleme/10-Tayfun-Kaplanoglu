# USAGE
# python train_mask_detector.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, Xception, VGG16, ResNet50V2	,NASNetMobile
from tensorflow.keras.layers import AveragePooling2D,GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.xception import preprocess_input

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
# https://keras.io/api/applications/
# https://www.kaggle.com/abnera/transfer-learning-keras-xception-cnn

class classifier:

	def __init__(self):
		self.folder_mask = "dataset/"
		self.weights = "weights.h5"
		self.model_out = "results/mask_detector_new.model"
		self.res_plot = "results/plot.png"
		self.classes = ["mask", "no_mask"]

		# initialize the initial learning rate, number of epochs to train for,
		# and batch size
		self.INIT_LR = 1e-4
		self.EPOCHS = 40
		self.BS = 32
		self.size = 224
		self.c = 0

	def load_data(self):
		# grab the list of images in our dataset directory, then initialize
		# the list of data (i.e., images) and class images
		print("[INFO] loading images...")
		imagePaths = list(paths.list_images(self.folder_mask))
		data = []
		labels = []

		# loop over the image paths
		for imagePath in imagePaths:
			# extract the class label from the filename
			label = imagePath.split(os.path.sep)[-2]

			# load the input image (224x224) and preprocess it
			#image = load_img(imagePath, target_size=(224, 224)) # return float image
			#image = img_to_array(image)
			#open_cv_image = image[:, :, ::-1].copy() 

			image = cv2.imread(imagePath,1)
			image = cv2.resize(image, (self.size,self.size))
			image = image.astype('float')
			image = image/255.0
			#image = preprocess_input(image)

			#cv2.imshow("jk",image)
			#cv2.waitKey(0)

			# update the data and labels lists, respectively
			data.append(image)
			labels.append(label)

		# convert the data and labels to NumPy arrays
		data = np.array(data, dtype="float32")
		labels = np.array(labels)

		# perform one-hot encoding on the labels
		lb = LabelBinarizer()
		labels = lb.fit_transform(labels)
		labels = to_categorical(labels)

		# partition the data into training and testing splits using 75% of
		# the data for training and the remaining 25% for testing
		(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
		return trainX, testX, trainY, testY

	def get_model(self):
		# load the MobileNetV2 network, ensuring the head FC layer sets are
		# left off
		baseModel = MobileNetV2(weights="imagenet", include_top=False,
			input_tensor=Input(shape=(self.size, self.size, 3)))

		# construct the head of the model that will be placed on top of the
		# the base model
		headModel = baseModel.output
		headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
		#headModel = GlobalAveragePooling2D()(headModel)
		headModel = Flatten(name="flatten")(headModel)
		headModel = Dense(512, activation="relu")(headModel)
		headModel = Dropout(0.5)(headModel)
		headModel = Dense(2, activation="softmax")(headModel)
		

		# place the head FC model on top of the base model (this will become
		# the actual model we will train)
		model = Model(inputs=baseModel.input, outputs=headModel)

		# loop over all layers in the base model and freeze them so they will
		# *not* be updated during the first training process
		for layer in baseModel.layers:
			layer.trainable = False

		# compile our model
		print("[INFO] compiling model...")
		opt = Adam(lr=self.INIT_LR) #, decay=self.INIT_LR / EPOCHS)
		model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["acc"])
		return model

	def train(self,trainX, testX, trainY, testY):

		model = self.get_model()
		# train the head of the network
		# Early stopping
		weight_saver = ModelCheckpoint(self.weights, monitor='val_acc', save_best_only=True, verbose=1, mode='max') # save_weights_only=True)   # save_best_only=true save the best val_acc weights

		print("[INFO] training head...")

		H = model.fit_generator(self.my_generator(trainX, trainY),
						epochs=self.EPOCHS, # 60 now 32 before
                        steps_per_epoch = len(trainX) // self.BS,
                        validation_data=self.my_generator(testX, testY),
                        validation_steps=len(testX) // self.BS,
                        verbose=1 ,#self.verbosity,
                        callbacks=[weight_saver] 
						)

		# plot the training loss and accuracy
		N = self.EPOCHS
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
		plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
		plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
		plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig(self.res_plot)

	def my_generator(self, trainX, trainY):
		SEED=42  
		# construct the training image generator for data augmentation
		aug = ImageDataGenerator(
			#brightness_range=(0.7,1.0), # NOT WORKING
			rotation_range=3,
			zoom_range=0.1,
			width_shift_range=0.1,
			shear_range=0.1,
			height_shift_range=0.1,
			horizontal_flip=True
		).flow(trainX, trainY, batch_size=self.BS,shuffle=True,seed=SEED)
		while True:
			images_aug_batch, y_aug_batch = aug.next()
			yield images_aug_batch , y_aug_batch

	def test_generator(self,x,y):
		x_train_batch, y_train_batch  =  next(self.my_generator(x, y))

		for i in range(0,self.BS):
			im =  x_train_batch[i,:,:,:]
			cv2.putText(im, str( y_train_batch[i].argmax()  ), (30,30),  cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,0), 1, cv2.LINE_AA  )
			cv2.imwrite("check/"+str(self.c)+".png",im)
			self.c = self.c+1

	def report(self,X,Y):
		model=self.get_model()
		model.load_weights(self.weights)
		#model.save(self.model_out, save_format="h5")

		# make predictions on the testing set
		print("[INFO] evaluating network...")
		predIdxs = model.predict(X) #, batch_size=self.BS)

		# for each image in the testing set we need to find the index of the
		# label with corresponding largest predicted probability
		predIdxs = np.argmax(predIdxs, axis=1)

		# show a nicely formatted classification report
		print(classification_report(Y.argmax(axis=1), predIdxs,target_names=self.classes))

		# serialize the model to disk
		#print("[INFO] saving mask detector model...")
		#model.save(self.model_out, save_format="h5")

	def save(self):
		model=self.get_model()
		model.load_weights(self.weights)
		model.save(self.model_out, save_format="h5")


cl = classifier()
trainX, testX, trainY, testY = cl.load_data()

#print("TEST GENERATOR")
cl.test_generator(trainX,trainY)

#print("[TRAIN")
cl.train(trainX, testX, trainY, testY)

#print("[TRAINING RESULT REPORT]")
cl.report(trainX,trainY)

#print("[TEST RESULT REPORT]")
cl.report(testX,testY)

#print("[SAVE]")
cl.save()
