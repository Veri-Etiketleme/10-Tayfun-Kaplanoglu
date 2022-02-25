from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing import image
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.misc import imread
import tensorflow as tf
from keras import backend as K
import math
import time
import cv2
from ssd_keras.ssd_v2 import SSD300v2
from ssd_keras.ssd_utils import BBoxUtility

thickness = 2 # bbox thickness

# config = tf.ConfigProto(
#     device_count={'GPU': 0}
# )
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

np.set_printoptions(suppress=True)

voc_classes = ['Aeroplane', 'Bicycle', 'Bird', 'Boat', 'Bottle',
               'Bus', 'Car', 'Cat', 'Chair', 'Cow', 'Diningtable',
               'Dog', 'Horse','Motorbike', 'Person', 'Pottedplant',
               'Sheep', 'Sofa', 'Train', 'Tvmonitor']
NUM_CLASSES = len(voc_classes) + 1

network_size = 300
input_shape=(network_size, network_size, 3)
model = SSD300v2(input_shape, num_classes=NUM_CLASSES)
model.load_weights('model_data/weights_SSD300.hdf5', by_name=True)
bbox_util = BBoxUtility(NUM_CLASSES)


first_frame_has_car = True
prev_bboxes = []
bbox_disappear_frame_count = 0
prev_bboxes_len = 0

def car_ssd(input_img):
    
    inputs = []
    #input_img_cropped = input_img[120:720,680:1280,:]
    #img = cv2.resize(input_img_cropped, (300, 300))
    img = cv2.resize(input_img, (300, 300))
    img = image.img_to_array(img)
    inputs.append(img.copy())
    inputs = preprocess_input(np.array(inputs))
    inputs = np.expand_dims(inputs[0], axis=0)
    
    preds = model.predict(inputs, batch_size=1, verbose=0)
    results = bbox_util.detection_out(preds)
    
    final_img, bboxes = draw_boxes(input_img, preds, results)
    bboxes = sorted(bboxes, key=lambda t:(t[2]-t[0])*(t[3]-t[1]), reverse=True)
    return final_img, bboxes


def get_bbox(xmin, ymin, xmax, ymax):
    return [xmin, ymin, xmax, ymax]

def center_is_near(prev_bbox, bbox):
    IS_NEAR_THRESHOLD = 30
    
    prev_center_x = (prev_bbox[0] + prev_bbox[2])/2.
    prev_center_y = (prev_bbox[1] + prev_bbox[3])/2.
    center_x = (bbox[0] + bbox[2])/2.
    center_y = (bbox[1] + bbox[3])/2.
    
    dist = np.sqrt((prev_center_x - center_x)**2 + (prev_center_y - center_y)**2)
    
    if dist <= IS_NEAR_THRESHOLD:
        return True
    else:
        return False

def draw_boxes(img, preds, results):
    global first_frame_has_car, prev_bboxes, prev_bboxes_len, bbox_disappear_frame_count
    
    # Parse the outputs.
    det_label = results[0][:, 0]
    det_conf = results[0][:, 1]
    det_xmin = results[0][:, 2]
    det_ymin = results[0][:, 3]
    det_xmax = results[0][:, 4]
    det_ymax = results[0][:, 5]

    # Get detections with confidence higher than 0.6.
    top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]

    top_conf = det_conf[top_indices]
    top_label_indices = det_label[top_indices].tolist()
    top_xmin = det_xmin[top_indices]
    top_ymin = det_ymin[top_indices]
    top_xmax = det_xmax[top_indices]
    top_ymax = det_ymax[top_indices]
    
    bboxes_len = 0
    bboxes = []   
    for i in range(top_conf.shape[0]):
        
        label = int(top_label_indices[i])        
        label_name = voc_classes[label - 1]
        
        if label_name == 'Car':
            bboxes_len += 1
            
            xmin = int(round(top_xmin[i] * img.shape[1]))
            ymin = int(round(top_ymin[i] * img.shape[0]))
            xmax = int(round(top_xmax[i] * img.shape[1]))
            ymax = int(round(top_ymax[i] * img.shape[0]))
            
            if first_frame_has_car or len(prev_bboxes) == 0:
                prev_bboxes.append(get_bbox(xmin, ymin, xmax, ymax))
                first_frame_has_car = False
                prev_bboxes_len = 0
            else:
                has_near_in_prev_bboxes = False
                for i_prev_bbox in range(len(prev_bboxes)):
                    if center_is_near(prev_bboxes[i_prev_bbox], [xmin, ymin, xmax, ymax]):
                        ratiox = 0.5
                        ratioy = 0.65
                        xmin = int((1-ratiox)*xmin + ratiox*prev_bboxes[i_prev_bbox][0])
                        ymin = int((1-ratioy)*ymin + ratioy*prev_bboxes[i_prev_bbox][1])
                        xmax = int((1-ratiox)*xmax + ratiox*prev_bboxes[i_prev_bbox][2])
                        ymax = int((1-ratioy)*ymax + ratioy*prev_bboxes[i_prev_bbox][3])
                        prev_bboxes[i_prev_bbox][0] = xmin 
                        prev_bboxes[i_prev_bbox][1] = ymin
                        prev_bboxes[i_prev_bbox][2] = xmax
                        prev_bboxes[i_prev_bbox][3] = ymax
                        has_near_in_prev_bboxes = True
                        
                if not has_near_in_prev_bboxes:
                    prev_bboxes.append(get_bbox(xmin, ymin, xmax, ymax))
                    
            bboxes.append(get_bbox(xmin, ymin, xmax, ymax))  
           
    if prev_bboxes_len > bboxes_len and bbox_disappear_frame_count < 5:
        for i_prev_bbox in range(len(prev_bboxes)):
            for i_bbox in range(len(bboxes)):
                if not center_is_near(prev_bboxes[i_prev_bbox], bboxes[i_bbox]):
                    cv2.rectangle(img, 
                                  (prev_bboxes[i_prev_bbox][0],prev_bboxes[i_prev_bbox][1]), 
                                  (prev_bboxes[i_prev_bbox][2],prev_bboxes[i_prev_bbox][3]), (0,255,0), thickness)
            if len(bboxes) == 0:
                cv2.rectangle(img, 
                              (prev_bboxes[i_prev_bbox][0],prev_bboxes[i_prev_bbox][1]), 
                              (prev_bboxes[i_prev_bbox][2],prev_bboxes[i_prev_bbox][3]), (0,255,0), thickness)
        bbox_disappear_frame_count += 1
    else:
        bbox_disappear_frame_count = 0
        prev_bboxes_len = len(bboxes)
        prev_bboxes = bboxes
    for i_bbox in range(len(bboxes)):
        cv2.rectangle(img, (bboxes[i_bbox][0],bboxes[i_bbox][1]), (bboxes[i_bbox][2],bboxes[i_bbox][3]), (0,255,0), thickness)  
            
    if len(prev_bboxes) > 10:
        prev_bboxes = []
        bbox_disappear_frame_count = 10
        
    return img, bboxes
