from keras.preprocessing.image import ImageDataGenerator  
import numpy as np
import cv2
import os
import shutil

# Data augmetnation: https://debuggercafe.com/image-augmentation-with-keras-for-better-deep-learning/

name = "no_glass"
folder_path = "dataset/"
folder_path_augmented = name + "_augmented"
try: 
    os.mkdir(path) 
except OSError as error: 
    print(error) 

IMG_SIZE = (32,32) # W, H
BATCH_SIZE = 500

def my_generator(batch_size):
    SEED=42  
    images_generator = ImageDataGenerator(
            #rescale=1./255,
            #brightness_range=(0.7, 0.9),
            vertical_flip=False,
            horizontal_flip=True,
            width_shift_range=0.12, #0.05,
            height_shift_range=0.12, #0.05,
            shear_range=5,
            rotation_range=10, #2,
            zoom_range=0.15).flow_from_directory(
                directory=folder_path,
                classes=[name],
                target_size=IMG_SIZE,
                color_mode="rgb",
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=True,
                seed=42
            )
        #.flow(logo, logo, batch_size, seed=SEED)
    while True:
        images_aug_batch,_ = images_generator.next()
        yield images_aug_batch  

for m in range(0,2):
    images_aug_batch  = next(my_generator(BATCH_SIZE))

    print(images_aug_batch.shape)
    for i in range(0,images_aug_batch.shape[0]):
        cv2.imwrite("datasets/"+name+"-augmented/" + str(m) + "_" + str(i) + ".png", images_aug_batch[i,:,:,0])
        cv2.waitKey(0)

