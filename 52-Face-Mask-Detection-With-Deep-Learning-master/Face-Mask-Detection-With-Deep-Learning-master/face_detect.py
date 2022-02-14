from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os


def detect_predict(frame,faceNet,maskNet):

    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

    
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    preds = []
    locs = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        print(confidence)

        if confidence*100 > 90  :

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))


            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:

        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return (locs, preds)


    


vs = VideoStream(src=0).start()
time.sleep(2.0)

## creating faceNet

faceNet = cv2.dnn.readNet('deploy.prototxt','res10_300x300_ssd_iter_140000.caffemodel')

## creating maskNet

maskNet = load_model('trained_model.h5')

while True:

    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    (locs,pred) = detect_predict(frame,faceNet,maskNet)

    for (box,preds) in zip(locs,pred):

        (startX , startY , endX , endY) = box
        (with_mask,without_mask)=preds

        if with_mask > without_mask:
            label = 'mask'
            color = (0,255,0)
        else:
            label = 'No mask'
            color = (255,0,0)

        label = "{}: {:.2f}%".format(label, max(with_mask, without_mask) * 100)

        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()   
vs.stop()