Update (10-10-2021) My email has been changed to yijingru321@gmail.com.
# BBAVectors-Oriented-Object-Detection
[WACV2021] Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors ([arXiv](https://arxiv.org/pdf/2008.07043.pdf))

Please cite the article in your publications if it helps your research:

	@inproceedings{yi2021oriented,
	title={Oriented object detection in aerial images with box boundary-aware vectors},
	author={Yi, Jingru and Wu, Pengxiang and Liu, Bo and Huang, Qiaoying and Qu, Hui and Metaxas, Dimitris},
	booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
	pages={2150--2159},
	year={2021}
	}


# Introduction

Oriented object detection in aerial images is a challenging task as the objects in aerial images are displayed in arbitrary directions and are usually densely packed. Current oriented object detection methods mainly rely on two-stage anchor-based detectors. However, the anchor-based detectors typically suffer from a severe imbalance issue between the positive and negative anchor boxes. To address this issue, in this work we extend the horizontal keypoint-based object detector to the oriented object detection task. In particular, we first detect the center keypoints of the objects, based on which we then regress the box boundary-aware vectors (BBAVectors) to capture the oriented bounding boxes. The box boundary-aware vectors are distributed in the four quadrants of a Cartesian coordinate system for all arbitrarily oriented objects. To relieve the difficulty of learning the vectors in the corner cases, we further classify the oriented bounding boxes into horizontal and rotational bounding boxes. In the experiment, we show that learning the box boundary-aware vectors is superior to directly predicting the width, height, and angle of an oriented bounding box, as adopted in the baseline method. Besides, the proposed method competes favorably with state-of-the-art methods.

<p align="center">
	<img src="imgs/img1.png", width="800">
</p>

# Evaluation Results on [DOTA-v1.0](https://captain-whu.github.io/DOTA/evaluation.html)

When training the BBAVectors+rh on 4 RTX6000 GPUs with a larger batch size```--batch_size 48```, we get a higher mAP (75.36) than the reported mAP (72.32) in the paper. We add the result to our final version. We thank the public visitors for their effort. The model weights can be downloaded from the following links: [GoogleDrive](https://drive.google.com/drive/folders/1a5LirNJ9-jc21JV11WBGqDYKpur95sno?usp=sharing) and [Dropbox](https://www.dropbox.com/sh/p7pz6silvy56f1a/AADHGlBKmdf5-7GBq2q7XBTua?dl=0).


```ruby
## model_50.pth
mAP: 0.7536283690546086
ap of each class: plane:0.8862514770737425, baseball-diamond:0.8406009896282075, bridge:0.521285610860641, ground-track-field:0.6955552280263699, small-vehicle:0.7825702607967113, large-vehicle:0.8040010247209182, ship:0.8805575982076236, tennis-court:0.9087489402165854, basketball-court:0.8722663525600673, storage-tank:0.8638699841268725, soccer-ball-field:0.5610545208583243, roundabout:0.6562139014619145, harbor:0.6709747110284013, swimming-pool:0.7208480121858474, helicopter:0.6396269240669054

## model_43.pth
mAP: 0.7492727335105831
ap of each class: plane:0.8859121197958046, baseball-diamond:0.8483251642688572, bridge:0.5214374843409882, ground-track-field:0.6560710395759289, small-vehicle:0.7773671634218439, large-vehicle:0.7427879633964128, ship:0.8804625721887132, tennis-court:0.908816372618596, basketball-court:0.862399364058993, storage-tank:0.8670730838290734, soccer-ball-field:0.5987801663737911, roundabout:0.6401450110418495, harbor:0.6698206063852568, swimming-pool:0.7071826121359568, helicopter:0.672510279226682
```


# Dependencies
Ubuntu 18.04, Python 3.6.10, PyTorch 1.6.0, OpenCV-Python 4.3.0.36 

# How To Start

Download and install the DOTA development kit [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) and put it under datasets folder.
Please uncomment the ```nn.BatchNorm2d(head_conv)``` in ```ctrbox_net.py``` to avoid ```NAN``` loss when training with a smaller batch size. Note that the current version of ```ctrbox_net.py``` matches the uploaded weights.

## About DOTA
### Split Image
Split the DOTA images from [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) before training, testing and evaluation.

The dota ```trainval``` and ```test``` datasets are cropped into ```600×600``` patches with a stride of `100` and two scales `0.5` and `1`. 

### About txt Files
The `trainval.txt` and `test.txt` used in `datasets/dataset_dota.py` contain the list of image names without suffix, example:
```
P0000__0.5__0___0
P0000__0.5__0___1000
P0000__0.5__0___1500
P0000__0.5__0___2000
P0000__0.5__0___2151
P0000__0.5__0___500
P0000__0.5__1000___0
```
Some people would be interested in the format of the ground-truth, I provide some examples for DOTA dataset:
Format: `x1, y1, x2, y2, x3, y3, x4, y4, category, difficulty`

Examples: 
```
275.0 463.0 411.0 587.0 312.0 600.0 222.0 532.0 tennis-court 0
341.0 376.0 487.0 487.0 434.0 556.0 287.0 444.0 tennis-court 0
428.0 6.0 519.0 66.0 492.0 108.0 405.0 50.0 bridge 0
```
## Data Arrangment
### DOTA
```
data_dir/
        images/*.png
        labelTxt/*.txt
        trainval.txt
        test.txt
```
you may modify `datasets/dataset_dota.py` to adapt code to your own data.
### HRSC
```
data_dir/
        AllImages/*.bmp
        Annotations/*.xml
        train.txt
        test.txt
        val.txt
```
you may modify `datasets/dataset_hrsc.py` to adapt code to your own data.


## Train Model
```ruby
python main.py --data_dir dataPath --epochs 80 --batch_size 16 --dataset dota --phase train
```

## Test Model
```ruby
python main.py --data_dir dataPath --batch_size 16 --dataset dota --phase test
```


## Evaluate Model
```ruby
python main.py --data_dir dataPath --conf_thresh 0.1 --batch_size 16 --dataset dota --phase eval
```

You may change `conf_thresh` to get a better `mAP`. 

Please zip and upload the generated `merge_dota` for DOTA [Task1](https://captain-whu.github.io/DOTA/evaluation.html) evaluation.
