# **Advanced Lane Lines detection**

This is the fourth project I am doing as part of Udacity's Self-Driving-Car Nanodegree.

**The goals/steps of this project are the following:**
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)

[image1]: ./output_images/final_result.gif "Final pipeline result"
[image2]: ./output_images/undistort_chessboard.png "Undistort chessboard example"
[image3]: ./output_images/undistort.png "Undistort example"
[image4]: ./output_images/binarized.png "Binarized Example"
[image5]: ./output_images/warped.png "Warped example"
[image6]: ./output_images/binarized_lane_detected.png "Binarized with lane example"
[image7]: ./output_images/histogram.png "Binarized histogram"
[image8]: ./output_images/0.png "Final example 1"
[image9]: ./output_images/1.png "Final example 2"

# Report
### Writeup & Project Files

#### 1. Writeup
You're reading it! I will examine below the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.


#### 2. Project Files

You can find all project files in this [Github Repository](https://github.com/thoomi/advanced-lane-line-detection).

My project includes the following files and folders:
* [src/](https://github.com/thoomi/advanced-lane-line-detection/tree/master/src) containing all coded script files including the whole processing pipeline
* [videos/](https://github.com/thoomi/advanced-lane-line-detection/tree/master/videos) containing final output videos of the **project_video**, **challenge_video** and **harder_challenge_video**
* [camera_cal/](https://github.com/thoomi/advanced-lane-line-detection/tree/master/camera_cal) containing camera calibration images
* [output_images/](https://github.com/thoomi/advanced-lane-line-detection/tree/master/output_images) containing pipeline images of each individual step
* [calibration_coefficients.p](https://github.com/thoomi/advanced-lane-line-detection/blob/master/camera_cal/calibration_coefficients.p) containing the calculated calibration coefficients and the calibration matrix
* [interactive_evaluation.ipynb](https://github.com/thoomi/advanced-lane-line-detection/blob/master/interactive_evaluation.ipynb) a IPython notebook to evaluate and test various aspects of the pipeline easier
* [writeup_report.md](https://github.com/thoomi/advanced-lane-line-detection/blob/master/writeup_report.md)

---

### Camera Calibration

#### 1.  Briefly state how you computed the camera matrix and distortion coefficients

The code for this step is contained in the file `CameraCalibrator.py`. It contains a class entirely dedicated to load chessboard images, process them and calculate the distortion coefficients and the distortion matrix.

The method `calibrate()` takes in a list of calibration images and the chessboard size to be found in the images.
It loads *object* and *image points* by utilizing the OpenCV function `cv2.findChessboardCorners()` and calculates the distortion parameters by using `cv2.calibrateCamera()`. It then saves the coefficients and matrix into a pickle file in order to allow it to reload them later. The class method `undistort()` takes in an image and returns the undistorted counterpart calculated with the previously obtained values.

Here is one of the calibration images to show the working undistortion:

![Example of undistorted image][image2]

---

### Lane-line Pipeline

#### 1. Undistortion

The frames undistortion will be performed in line 40 of the file `process.py` where the `undistort()` method of the `CameraCalibrator` is called. The CameraCalibrator loads the previously obtained coefficients and the matrix to do this step. See the example image below:

![Example of undistorted image][image3]


#### 2. Binarization and Thresholding

In line 42 of the `process.py` file the binarization and thresholding is done. The line calls the process method of the `Binarizer` class. In there, the processing takes place within the lines 15 and 35. I first convert the image into YUV color space and make use of a combination of the Y and V channels. Additionally, a Sobel operator is run in both x and y-direction, and put together by a binary OR. There is an example image below:

![Example of binarized image][image4]


#### 3. Perspective Transformation

The class `Warper` is responsible for the perspective transformation of the binarized image. This is done in order to detect the lane line pixels easier. The Transformation-Matrix is calculated by the OpenCV function `cv2.getPerspectiveTransform()` and is defined by the the points in the following table and calculated within the lines 10-34 in the file `Warper.py`.

| Source        | Destination   |
|:-------------:|:-------------:|
| 570, 470      | 490, 0        |
| 720, 470      | 1020, 0       |
| 1120, 720     | 1020, 720     |
| 190, 720      | 490, 720      |

An example of the binarized and warped image:

![Example of undistorted image][image5]


#### 4. Lane-line Pixel Identification

The code for this whole process can be found in the file `Lane.py`.

The identification is done by driving the image into rows and slide a window within a row in the left and right part of the image in order to identify hot pixels related to a lane-line. The starting point for the left and right window is determined by finding peaks within a histogram of the image. For example, the points for starting the sliding window search on the image below are identified as (500, 0) for the left and (1020, 0) for the right image. This functionality can be found within the lines 132-184 of the `Lane.py` file.

![Example of histogram image][image7]

Have the pixels of the line been identified a polynomial is fitted for each line. ,  happens in line 42 of the `Line.py` file. Using this polynomial from a previous frame, all successive line searches can be done along this polynomial because it is very unlikely the lines change radically between two frames. Below is an example with identified lane-lines and the corresponding polynomial areas:

![Example of undistorted image][image6]


#### 5. Calculation of lane curvature radius and position of vehicle

The calculation of the lanes radius of curvature is done within the lines 71-74 of the `Lane.py` file. First, the polynomial of the lane center is calculated and then the methods `caluculate_lane_curvature()` and `Calculate_center_offset()` are called for the actual calculation.

#### 6. Final Result & Plotted Lane Area
Here is a picture of the final output. The identified lane-line is plotted onto the original image and the radius of curvature and the center offset are displayed in the upper left. Additionally, there are two smaller images (in the upper right) from within the pipeline get a better understanding of how it works internally.

![Example of final image][image8]


#### 7. Further Pipeline Features
In order to create a more robust pipeline, I implemented some additional features which follow below in more detail.

- **Lane Smoothing** (Line.py lines 46-49)
    - In order to prevent the detected lane from jittering if the lines have not been detected correctly for a frame, the class Line stores the last 7 detected lane-lines and calculates the average.
- **Sanity Check** (Lane.py lines 77-100)
    - This feature checks whether the lines detected within a frame are plausible. It is a combination of three checks. At first, it is checked if there are line pixels at all. After that the detected left and right lines are verified to be parallel within a given variance. The third test checks if the lines are plausible based on their horizontal distance.
2. **Frame Dropping** (Lane.py lines 54-59)
    - A frame which does not pass the sanity check will be dropped. If this happens for multiple frames in a row the previous Lines will be reset and the detection process starts again with the histogram based search.

---


### Result video
[videothumb1]: ./output_images/0.png "Final example 1"
[videothumb2]: ./output_images/4.png "Final example 2"


#### 1. Project Video Result

[![Project video thumbnail][videothumb1]](./videos/result_project_video.mp4?raw=true)


#### 2. Challenge Video Result
[![Project video thumbnail][videothumb2]](./videos/result_challenge_video.mp4?raw=true)


---


### Discussion & Reflection

I had a hard time figuring out a good combination in order to create the binary image. That is why I created the IPython notebook to allow an easier process for finding good parameters and thresholds. One of the challenges was to find a good way to check if two detected lines are plausible and what to do if they are not. After all, the pipeline will fail if the binary image is too noisy due to bright lighting or in very sharp curves where it is hard to detect any line at all.

In a nutshell, this project took quite a lot of time figuring out the correct parameters for each step in the pipeline and to smooth out the detected lane at the end. But I still had a lot of fun and learned a lot about OpenCV and computer vision problems in general.


---

### Appendix


#### Blogs & Tutorials
[blog1]: https://medium.com/towards-data-science/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3
[blog2]: https://github.com/pkern90/CarND-advancedLaneLines

- [Robust lane finding using advanced computer vision techniques (Vivek Yadav)][blog1]
- [Advanced Lane Finding (Patrick Kern)][blog2]

#### Tools:
[tool01]: https://www.python.org/
[tool02]: http://opencv.org/

 - [Python][tool01]
 - [OpenCV][tool02]
