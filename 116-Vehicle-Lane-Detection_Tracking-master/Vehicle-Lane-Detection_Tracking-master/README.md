## Vehicle and Lane-line Detection and Tracking

#### SSD
[![project-ssd](./output_videos/project_video_ssd.gif)](https://youtu.be/jVCQNJfLMHE) [![challenge-ssd](./output_videos/challenge_video_ssd.gif)](https://youtu.be/AyCdamAjrW8)

#### Yolo3
[![project-yolo3](./output_videos/project_video_yolo3.gif)](https://youtu.be/_d-ocTv7jd8) [![challenge-yolo3](./output_videos/challenge_video_yolo3.gif)](https://youtu.be/O2VmbVHWcyw)

#### MRCNN
[![project-mrcnn](./output_videos/project_video_mrcnn.gif)](https://youtu.be/la52PyQrX-o) [![challenge-mrcnn](./output_videos/challenge_video_mrcnn.gif)](https://youtu.be/Xq-aRmKeT8g)


---

### Overview

Combine the lane finding and vehicle detection projects together.

Add the car class to track the location (bounding box) and history of the vehicles detected. 

The perspective transformation to the bird's-eye view that cover the lanes along the camera direction is used to measure the x and y displacement to the camera.


Results using SSD, Yolo3 and Mask R-CNN models are presented.

The relative distance (dx, dy) in unit of meter is shown above the bounding box of the detected car.

The relative speed (vx, vy) in unit of kilometer/hour is shown below the bounding box.

The thumbnail and distance / speed of detected vehicles are also available in the top-left frame of the video.
The vehicles are sorted by the size of bounding box in descending order.






[//]: # (Image References)
[image1]: ./ref/kitti.png
[image2]: ./ref/gti_far.png
[image3]: ./ref/gti_middle.png
[image4]: ./ref/gti_left.png
[image5]: ./ref/gti_right.png
[image6]: ./ref/extra_non0.png
[image7]: ./ref/extra_non1.png
[image8]: ./ref/extra_non2.png
[image9]: ./ref/gti_non0.png
[image10]: ./ref/gti_non1.png
[image11]: ./ref/hog_car.png
[image12]: ./ref/hog_notcar.png
[image13]: ./ref/C_graph.png
[image14]: ./ref/sliding_window.png
[image15]: ./output_images/output_images.png
[video1]: ./output_images/project_video_svm.gif
[video2]: ./output_images/project_video_mrcnn.gif


---


### Dataset

The project dataset is provided by Udacity. It is split into [vehicles images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles images](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). The dataset is a combination of KITTI vision benchmark suite and GTI vehicle image database. GTI vehicle images are grouped into far, left, right, middleClose.

These are examples of cars and non cars:


| KITTI     | GTI Far     | GTI Close  | GTI Left    | GTI Right  |
|-----------|-------------|------------|-------------|------------|
|![][image1]| ![][image2] | ![][image3]| ![][image4] |![][image5] |
| Non car1  | Non car2    | Non Car3   | GTI Non car1| GTI Non car2|
|![][image6]| ![][image7] | ![][image8]| ![][image9] |![][image10]|





### Histogram of Oriented Gradients (HOG)

#### 1. Explain how to extract HOG features from the training images.

In this project, Support Vector Machines (SVM) are used to train a model to classify if an image contains a car or not. The Histogram of Oriented Gradients (HOG) are used as the feature representation.

I started by reading in all the `vehicle` and `non-vehicle` images, then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here are examples using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(1, 1)`:

Samples of images in the car dataset and the corresponding HOG graphs -
![alt text][image11]

Samples of images in the not-car dataset and the corresponding HOG graphs -
![alt text][image12]

The code for this step is contained in lines 263 through 351 of the file [`util_svm.py`](util_svm.py).  



#### 2. Explain how to settle on final choice of HOG parameters.

I tried various combinations of parameters and came up with this parameter set -

```python
### Parameters ###
color_space    = 'YCrCb'    # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient         = 9          # HOG orientations
pix_per_cell   = 8          # HOG pixels per cell
cell_per_block = 1          # HOG cells per block, which can handel e.g. shadows
hog_channel    = 0          # Can be 0, 1, 2, or 'ALL'
spatial_size   = (16, 16)   # Spatial binning dimensions
hist_bins      = 32         # Number of histogram bins
spatial_feat   = True       # Spatial features on or off
hist_feat      = True       # Histogram features on or off
hog_feat       = True       # HOG features on or off
```

I only use channel Y of the YCrCb color space and set cell_per_block to 1. 
The feature vector length is 1440. This size is small enough for quick training and prediction, while maintaining all the important features  to get good accuracy. 

If I use full channels of YCrCb and cell_per_block as 2, the feature vector would be 6x larger, which takes 6 times longer to run, with less than 0.5% improvement of accuracy.


#### 3. Describe how to train a classifier using selected HOG features and color features.

I trained a linear SVM using `LinearSVC` in `sklearn.svm`. The training set is scaled to zero mean and unit variance before training the classifier, using `StandardScaler` in `sklearn.preprocessing`. The training code is in lines 563 through 591 of the file [`util_svm.py`](util_svm.py).

I further tuned the hyperparameter C in `LinearSVC` by doing a grid search and plotting the graph of C vs accuracy. C is the cost of classification in SVC (support vector classifier).

![alt text][image13]

Based on the graph above, `C = 0.0013` was chosen for `LinearSVC` function.

After the training, the test accuracy of SVC is `98.87 %`.

### Sliding Window Search

#### 1. Describe how to implement a sliding window search. How to decide what scales to search and how much to overlap windows.

I used 3 different sizes (`128x128`, `96x96`, `80x80`) of sliding windows starting at different locations, with `50 %` overlap. 

The blue boxes below are the sliding windows.

![alt text][image14]


I tried different combination of window sizes and locations for the sliding window search, checked the result and runtime, to decide on the final setup. 

The code of the sliding window search is in line 391 through 460 of file `util_svm`.



#### 2. Show some examples of test images to demonstrate how the pipeline is working.  What did I do to optimize the performance of the classifier?

Ultimately, I searched on three scales using YCrCb 1-channel (Y) HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image15]
---

### Video Implementation

#### 1. Provide a link to the final video output.  Pipeline performs reasonably well on the entire project video.
Here's a [link to test_video_svm.mp4](./output_images/test_video_svm.mp4).

Here's a [link to project_video_svm.mp4](./output_images/project_video_svm.mp4).

The project video using SVM is the animated GIF (1,1) - the upper-left video at the top of this document.


#### 2. Describe how to implement some kind of filter for false positives and some method for combining overlapping bounding boxes.

The pipeline to detect and draw bounding boxes of vehicle is the function `car_svm(image)` in line 185 through 208 of file `util_svm`.

The sliding windows that match the vehicle features are hot boxes. Heat map and bounding boxes of the vehicles can be generated from the hot boxes. 

The class AverageHotBox combines the overlapping hot boxes to create the average hot box, the strength (number of joined boxes) of the averageHotBox is used as filter for false positives. 

The code to combine and filter bounding boxes are in line 40 through 180 of file [`util_svm.py`](util_svm.py).


---

### Other models and methods of vehicle detection

In addition to HOG-SVM, I also apply other methods of vehicle detection to this project, based on the following models - Mask R-CNN, SSD and YOLOv3.

I use pre-trained model weights and treat car/bus/truck classes as vehicle, all other classes as non-vehicle.

#### 1. [Mask R-CNN](https://arxiv.org/abs/1703.06870)

* Interface - [`util_mrcnn.py`](./util_mrcnn.py)

* Reference - [Mask_RCNN](https://github.com/matterport/Mask_RCNN)

* Weight file - [mask_rcnn_coco.h5](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)

* Videos - [project_video_mrcnn.mp4](./output_images/project_video_mrcnn.mp4) /  [test_video_mrcnn.mp4](./output_images/test_video_mrcnn.mp4)

The project video using MR-CNN is the animated GIF (1,2) - the upper-right video at the top of this document.


#### 2. [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325)

* Interface - [`util_ssd.py`](./util_ssd.py)

* Reference - [ssd_keras](https://github.com/cory8249/ssd_keras)

* Weight file - [weights_SSD300.hdf5](https://mega.nz/#F!7RowVLCL!q3cEVRK9jyOSB9el3SssIA)

* Videos - [project_video_ssd.mp4](./output_images/project_video_ssd.mp4) /  [test_video_ssd.mp4](./output_images/test_video_ssd.mp4)

The project video using SSD is the animated GIF (2,1) - the lower-left video at the top of this document.



#### 3. [YOLOv3: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)
* Interface - [`util_yolo3.py`](./util_yolo3.py)

* Reference - [keras-yolo3](https://github.com/qqwweee/keras-yolo3)

* Weight file - convert from [YOLOv3-320.weights](https://pjreddie.com/media/files/yolov3.weights)

* Videos - [project_video_yolo3.mp4](./output_images/project_video_yolo3.mp4) /  [test_video_yolo3.mp4](./output_images/test_video_yolo3.mp4)

The project video using YOLOv3 is the animated GIF (2,2) - the lower-right video at the top of this document.





---

### Discussion

#### 1. The problems / issues with HOG-SVM.  Where the pipeline likely to fail, and how to make it more robust.

The issues I have with the HOG-SVM method are -

* Sliding window is slow and not flexible.
* Using threshold to filter out false positive is not robust enough to handle different situations.
* Averaging the video frames is needed to get stable result of the vehicle bounding box.

These are the main issues that lead me to try out other solutions. It turns out the deep learning approaches do not suffer from the problems mentioned above, and in general get better results compare to HOG-SVM.



#### 2. Performance and accuracy comparison

The numbers below are just rough estimates using my i7 CPU + 1070ti GPU desktop.

FPS (Frames Per Second) is the speed to process the video frames.

Process Time is the time it takes to process the 50 seconds project video. It is closely related to FPS.

|Method      | SVM         | MRCNN      | SSD-300     | YOLOv3     |
|------------|-------------|------------|-------------|------------|
|FPS         | 3.3         | 2          | 35          |13          |
|Process Time| 6 mins      | 10 mins    | 0.5 mins    |1.5 mins    |

The results of these 4 methods are shown at the top of this document.

While MRCNN takes the longest of processing time, it can identify the incoming vehicles from the opposite direction, and very small vehicles far in front of it. 

SSD is the fastest method. While it's known to have issues to identify small objects, it still outperforms SVM, generates very stable and accurate result.

YOLOv3 is very fast and can identify small objects very well. It outperforms YOLOv2 (which I also tested on this project) in both speed and accuracy. I expect the small or tiny version of YOLOv3 can compete with SSD-300 on the speed.



