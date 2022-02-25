import cv2
import glob
import pickle
import numpy as np
import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label
#from os import walk
from os import path
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from copy import copy
#from timeit import default_timer as timer

### Parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # 16 # 8 # HOG pixels per cell
cell_per_block = 1 # 2 # HOG cells per block, which can handel e.g. shadows
hog_channel = 0 #'ALL' # Can be 0, 1, 2, or 'ALL'
spatial_size = (16, 16) # (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # True # Spatial features on or off
hist_feat = True # True # Histogram features on or off
hog_feat = True # HOG features on or off
clf_path = 'car_svc_pickle.p'  # if classifier exist

Heatmap_buffer = []
N_buffer = 10 #3
buffer_weights=np.arange(1,N_buffer+1)/N_buffer

xy_windows = [(128, 128), (96, 96), (80, 80)]
xy_overlaps= [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
x_start_stops = [[None, None], [32, None], [400, 1280]]
y_start_stops = [[400, 650], [400, 650], [390, 550]]


# y_start_stop = [400, 656] # Min and max in y to search in slide_window()
# ystart_0 = y_start_stop[0]
# ystop_0 = ystart_0 + 64*2
# ystart_1 = ystart_0
# ystop_1 = y_start_stop[1]
# ystart_2 = ystart_0
# ystop_2 = y_start_stop[1]
# ystarts = [ystart_1, ystart_2]
# ystops = [ystop_1-100, ystop_2]
# search_window_scales = [1.5, 2]  # (64x64), (96x96), (128x128)

class AverageHotBox ():
    """Class that covers joining hot boxes algorithm.
    Idea is take fist box (called average box) form input boxes
    and join it with all boxes that are close enough.
    After joining two boxes we need to update average box (here
    just increasing size to cover both joining boxes).
    Loop until cannot further join.
    For left boxes repeat all procedure.
    As a result we also get average boxes strengths - the number of
    boxes it was joined to.
    This class represents one average box.
    """
    def __init__ (self, box):
        self.avg_box = [list(p) for p in box]
        self.detected_count = 1
        self.boxes = [box]
    
    def get_strength (self):
        """Returns number of joined boxes"""
        return self.detected_count
    
    def get_box (self):
        """Uses joined boxes information to compute
        this average box representation as hot box.
        This box has average center of all boxes and have
        size of 2 standard deviation by x and y coordinates of its points
        """
        if len(self.boxes) > 1:
            center = np.average (np.average (self.boxes, axis=1), axis=0).astype(np.int32).tolist()

            # getting all x and y coordinates of
            # all corners of joined boxes separately
            xs = np.array(self.boxes) [:,:,0]
            ys = np.array(self.boxes) [:,:,1]

            half_width = int(np.std (xs))
            half_height = int(np.std (ys))
            return (
                (
                    center[0] - half_width,
                    center[1] - half_height
                ), (
                    center[0] + half_width,
                    center[1] + half_height
                ))
        else:
            return self.boxes [0]
    
    def is_close (self, box):
        """Check wether specified box is close enough for joining
        to be close need to overlap by 30% of area of this box or the average box
        """
        # Reference -
        # http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
        x11 = self.avg_box [0][0]
        y11 = self.avg_box [0][1]
        x12 = self.avg_box [1][0]
        y12 = self.avg_box [1][1]
        x21 = box [0][0]
        y21 = box [0][1]
        x22 = box [1][0]
        y22 = box [1][1]
            
        x_overlap = max(0, min(x12,x22) - max(x11,x21))
        y_overlap = max(0, min(y12,y22) - max(y11,y21))

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)
        intersection = x_overlap * y_overlap;
        
        if (
            intersection >= 0.3 * area1 or
            intersection >= 0.3 * area2
        ):
            return True
        else:
            return False
    
    def join (self, boxes):
        """Join in all boxes from list of given boxes,
        removes joined boxes from input list of boxes
        """
        
        joined = False
        
        for b in boxes:
            if self.is_close (b):
                boxes.remove (b)
                self.boxes.append (b)
                self.detected_count += 1
                
                self.avg_box [0][0] = min (self.avg_box [0][0], b [0][0])
                self.avg_box [0][1] = min (self.avg_box [0][1], b [0][1])
                self.avg_box [1][0] = max (self.avg_box [1][0], b [1][0])
                self.avg_box [1][1] = max (self.avg_box [1][1], b [1][1])
                
                joined = True

        return joined

def calc_average_boxes (hot_boxes, strength):
    """Compute average boxes from specified hot boxes and returns
    average boxes with equals or higher strength
    """
    avg_boxes = []
    while len(hot_boxes) > 0:
        b = hot_boxes.pop (0)
        hb = AverageHotBox (b)
        while hb.join (hot_boxes):
            pass
        avg_boxes.append (hb)
    
    boxes = []
    for ab in avg_boxes:
        if ab.get_strength () >= strength:
            boxes.append (ab.get_box ())
    return boxes


class LastHotBoxesQueue ():
    """Class for accumulation of hot boxes from last 10 frames
    """
    def __init__ (self):
        self.queue_max_len = 10 # number items to store
        self.last_boxes = []

    def put_hot_boxes (self, boxes):
        """Put frame hot boxes
        """
        if (len(self.last_boxes) > self.queue_max_len):
            tmp = self.last_boxes.pop (0)
        
        self.last_boxes.append (boxes)
        
    def get_hot_boxes (self):
        """Get last 10 frames hot boxes
        """
        b = []
        for boxes in self.last_boxes:
            b.extend (boxes)
        return b


last_hot_boxes = LastHotBoxesQueue()
# heat_map = deque(np.array([np.zeros(img_size).astype(np.float)]), maxlen=N_buffer)
def car_svm(image_in): #, img_lane_augmented, lane_info):

    #start = timer()
    image = np.copy(image_in)
    img = image.astype(np.float32)/255

    windows_list, img_box = hot_windows(img)


    last_hot_boxes.put_hot_boxes(windows_list)
    hot_boxes = last_hot_boxes.get_hot_boxes ()
    
    # calculating average boxes and use strong ones
    # need to tune strength on particular classifer
    avg_boxes = calc_average_boxes (hot_boxes, 12) # 20)
    # heatmap = generate_heatmap(img, avg_boxes)
    # labels = label(heatmap)
    # cv2.putText(annotated_image1,"Close Vehicles: " + "{:0.0f}".format(labels[1]), org=(50,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #     fontScale=1, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    # image_with_boxes = draw_labeled_bboxes(img, labels)

    image_with_boxes = draw_boxes(img, avg_boxes, color=(0, 0, 1), thick=4)
    bboxes=[]
    for ab in avg_boxes:
        ((x1,y1),(x2,y2)) = ab  
        bboxes.append([x1,y1,x2,y2])
    return image_with_boxes * 255, bboxes

    # for search_window_scale, ystart, ystop in zip(search_window_scales, ystarts, ystops):
    #     windows_list_tmp = find_cars(np.copy(image), ystart, ystop, search_window_scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
    #                         spatial_size, hist_bins, hog_channel, color_space, spatial_feat, hist_feat, hog_feat)
    #     windows_list.extend(windows_list_tmp)

    # heatmap_pre = generate_heatmap(img, windows_list)
    # draw_img, heatmap_post, bboxes = draw_bboxes(image, copy(Heatmap_buffer), heatmap_pre, min(len(Heatmap_buffer)+1,N_buffer) )

    # if len(Heatmap_buffer) >= N_buffer:
    #     Heatmap_buffer.pop(0)

    # return draw_img


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        min_x = np.min(nonzerox)
        min_y = np.min(nonzeroy)
        max_x = np.max(nonzerox)
        max_y = np.max(nonzeroy)
        if (max_x - min_x) > 30 or (max_y - max_y)> 30:
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1],(0,0,255), 6)
            cv2.putText(img,"Vehicle: " + "{:0.0f}".format(car_number), org=(bbox[0][0],bbox[0][1]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.7, color=(255,255,255), lineType = cv2.LINE_AA, thickness=2)
    # return the image
    return img


# Define the hot windows by slide window search
# Returns: hot_windows(list), image_with_hot_windows_drawn(numpy.array)
def hot_windows(image):
    dst = np.copy (image)
    all_hot_wins = []
    
    # iterate over previousely defined sliding windows
    for x_limits, y_limits, window_size, overlap in zip(x_start_stops, y_start_stops, xy_windows, xy_overlaps):
        windows = slide_window( dst,
            x_start_stop=x_limits,
            y_start_stop=y_limits, 
            xy_window=window_size,
            xy_overlap=overlap)

        hot_wins = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                    spatial_size=spatial_size, hist_bins=hist_bins, 
                    orient=orient, pix_per_cell=pix_per_cell, 
                    cell_per_block=cell_per_block, 
                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                    hist_feat=hist_feat, hog_feat=hog_feat)                       
        
        all_hot_wins.extend(hot_wins)
        dst = draw_boxes(dst, hot_wins, color=(0, 0, 1), thick=4)

    return all_hot_wins, dst



# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# convert from RGB to another color space
def convert_color(img, color='YCrCb'):
    if color != 'RGB':
        image = cv2.cvtColor(img, eval('cv2.COLOR_RGB2'+color))
    else: image = np.copy(img)
    return image  

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []

        # png is scale from (0,1)
        image = mpimg.imread(file)

        # apply color conversion if other than 'RGB'
        feature_image = convert_color(image, color_space)  

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel],
                                        orient, pix_per_cell, cell_per_block,
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step)
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color(img, color_space)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32, orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))  # training image is (64,64)

        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        X = np.array(features).reshape(1, -1)
        test_features = scaler.transform(X)
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins,
    hog_channel, color_space, spatial_feat, hist_feat, hog_feat): # , dec_func_thresh=1.0):

    on_windows = []
    features = []
    img = img.astype(np.float32)/255

    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color=color_space)


    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above, hold the number of hog cells
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step: there are 8 cells, and move 2 cells per step, 75% overlap
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            if hog_feat:
                # Extract HOG for this patch
                if hog_channel == 0:
                    hog_features = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                elif hog_channel == 1:
                    hog_features = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                elif hog_channel == 2:
                    hog_features = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                else:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))


            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            features.append(hog_features)
            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))

            # Get color features
            if spatial_feat:
                spatial_features = bin_spatial(subimg, size=spatial_size)
                features.append(spatial_feat)

            if hist_feat:
                hist_features = color_hist(subimg, nbins=hist_bins)
                features.append(hist_features)

            #X = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            X = np.array(np.concatenate(features)).reshape(1, -1)
            
            test_features = X_scaler.transform(X)
            test_prediction = svc.predict(test_features)
            # test_dec_func = svc.decision_function(test_features)

            if test_prediction == 1 : # and test_dec_func >= dec_func_thresh :
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6)
                on_windows.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))

    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_bboxes(img, heatmap_buffer, heatmap_pre, N_buffer):

    heatmap_buffer.append(heatmap_pre)

    if len(heatmap_buffer) > N_buffer: # remove the first component if it is more than N_buffer elements
        heatmap_buffer.pop(0)

    # weight the heatmap based on current frame and previous N frames
    idxs = range(N_buffer)
    for b, w, idx in zip(heatmap_buffer, buffer_weights, idxs):
        heatmap_buffer[idx] = b * w

    heatmap = np.sum(np.array(heatmap_buffer), axis=0)
    heatmap = apply_threshold( heatmap, threshold= sum(buffer_weights[0:N_buffer])*2)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    bboxes = []
    # locate the bounding box
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox_tmp = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        bboxes.append(bbox_tmp)


    for bbox in bboxes:
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 4)

    # Return the image

    return img, heatmap, bboxes


def generate_heatmap(image, windows_list):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, windows_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # if np.amax(heatmap) == 0:
    #     flag = False
    # else:
    #     flag = True

    return heatmap



# if svm classifer exist, load it; otherwise, compute the svm classifier
if path.isfile(clf_path):

    print('loading existing classifier...')
    with open(clf_path, 'rb') as file:
        clf_pickle = pickle.load(file)
        svc = clf_pickle["svc"]
        X_scaler = clf_pickle["scaler"]
        orient = clf_pickle["orient"]
        pix_per_cell = clf_pickle["pix_per_cell"]
        cell_per_block = clf_pickle["cell_per_block"]
        spatial_size = clf_pickle["spatial_size"]
        hist_bins = clf_pickle["hist_bins"]
        color_space = clf_pickle["color_space"]

        hog_channel = clf_pickle["hog_channel"] 
        spatial_feat = clf_pickle["spatial_feat"]
        hist_feat = clf_pickle["hist_feat"]
        hog_feat = clf_pickle["hog_feat"] 
else:
    # Read in cars and notcars
    cars = glob.glob('train_images/vehicles/**/*.png')
    notcars = glob.glob('train_images/non-vehicles/**/*.png')

    # set the sample size
    sample_size = min(len(cars), len(notcars))
    cars = cars[0:sample_size]
    notcars = notcars[0:sample_size]

    print('Parameter file', clf_path, 'does not exist, train the car model...')

    print('extracting car features...')
    car_features = extract_features(cars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    print('car features extracted!')
    print('extracting noncar features...')
    notcar_features = extract_features(notcars, color_space=color_space,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=spatial_feat,
                            hist_feat=hist_feat, hog_feat=hog_feat)
    print('noncar features extracted!')

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    # rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=40)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc =  LinearSVC(C=0.004) # , loss='hinge')
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()

    # save classifier
    clf_pickle = {}
    clf_pickle["svc"] = svc
    clf_pickle["scaler"] = X_scaler
    clf_pickle["orient"] = orient
    clf_pickle["pix_per_cell"] = pix_per_cell
    clf_pickle["cell_per_block"] = cell_per_block
    clf_pickle["spatial_size"] = spatial_size
    clf_pickle["hist_bins"] = hist_bins
    clf_pickle["color_space"] = color_space

    clf_pickle["hog_channel"] = hog_channel
    clf_pickle["spatial_feat"] = spatial_feat
    clf_pickle["hist_feat"] = hist_feat
    clf_pickle["hog_feat"] = hog_feat

    pickle.dump(clf_pickle, open(clf_path, "wb"))
    print("Classifier is written into: {}".format(clf_path))

print('svc = ',svc)
print('scaler = ',X_scaler)
print('orient = ',orient)
print('pix_per_cell = ',pix_per_cell)
print('cell_per_block = ',cell_per_block)
print('spatial_size = ',spatial_size)
print('hist_bins = ',hist_bins)
print('color_space = ',color_space)
print('hog_channel = ',hog_channel)
print('spatial_feat = ',spatial_feat)
print('hist_feat = ',hist_feat)
print('hog_feat = ',hog_feat)





