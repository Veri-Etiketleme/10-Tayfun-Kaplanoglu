import matplotlib.pyplot as plt
import numpy as np
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras import layers
from keras import regularizers



xtrain , train_labels = tfds.as_numpy(tfds.load('cifar10',split='train',batch_size=-1,as_supervised=True,))
xtest , test_labels = tfds.as_numpy(tfds.load('cifar10',split='test',batch_size=-1,as_supervised=True,))
xtrain = (xtrain.astype('float32'))/255
xtest = (xtest.astype('float32'))/255
print(f"Train Shape: {xtrain.shape},Test Shape: {xtest.shape}")


input = keras.Input(shape=(32, 32, 3))

model = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input)
model = layers.MaxPooling2D((2, 2), padding='same')(model)
model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
model = layers.MaxPooling2D((2, 2), padding='same')(model)
model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
encoded = layers.MaxPooling2D((2, 2), padding='same')(model)

model=layers.Flatten()
model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
model = layers.UpSampling2D((2, 2))(model)
model = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(model)
model = layers.UpSampling2D((2, 2))(model)
model = layers.Conv2D(16, (3, 3), activation='relu',padding='same')(model)
model = layers.UpSampling2D((2, 2))(model)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(model)

autoencoder = keras.Model(input, decoded)
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy'])
autoencoder.summary()

history= autoencoder.fit(xtrain,xtrain,
                epochs=1000,
                batch_size=100,
                shuffle=True,
                validation_data=(xtest,xtest),
                callbacks=None, verbose=2)

                
f = plt.figure(figsize=(10,7))
f.add_subplot()

#Adding Subplot
plt.plot(history.epoch, history.history['loss'], label = "loss") # Loss curve for training set
plt.plot(history.epoch, history.history['val_loss'], label = "val_loss") # Loss curve for validation set

plt.title("Loss Curve",fontsize=18)
plt.xlabel("Epochs",fontsize=15)
plt.ylabel("Loss",fontsize=15)
plt.grid(alpha=0.3)
plt.legend()
plt.savefig("Test2_loss.png")
plt.show()



num_imgs = 48
rand = np.random.randint(1, xtest.shape[0]-48) 

xtestsample = xtest[rand:rand+num_imgs] # slicing
xvae = autoencoder.predict(xtestsample)
rows = 4 # defining no. of rows in figure
cols = 12 # defining no. of colums in figure
cell_size = 1.5
f = plt.figure(figsize=(cell_size*cols,cell_size*rows*2)) # defining a figure 
f.tight_layout()
for i in range(rows):
    for j in range(cols): 
        f.add_subplot(rows*2,cols, (2*i*cols)+(j+1)) # adding sub plot to figure on each iteration
        plt.imshow(xtestsample[i*cols + j]) 
        plt.axis("off")
        
    for j in range(cols): 
        f.add_subplot(rows*2,cols,((2*i+1)*cols)+(j+1)) # adding sub plot to figure on each iteration
        plt.imshow(xvae[i*cols + j]) 
        plt.axis("off")

f.suptitle("Autoencoder Results - Cifar10",fontsize=18)
plt.savefig("Test2.png")

plt.show()