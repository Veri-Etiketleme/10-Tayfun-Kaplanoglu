import argparse
import os
import sys
import cv2
import pickle
import numpy as np
import util_cal
import util_pipe
import util_lane
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from moviepy.editor import VideoFileClip

from settings import PERSPECTIVE_FILE_NAME, UNWARPED_SIZE
with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f)

warped_size = UNWARPED_SIZE
transform_matrix = perspective_data["perspective_transform"]
pix_per_meter = perspective_data['pixels_per_meter']


class FilterAvg:
    def __init__(self, vector, b, a):
        self.len = len(vector)
        self.b = b.reshape(-1, 1)
        self.a = a.reshape(-1, 1)
        vector = np.array(vector, dtype=np.float64)
        self.input_history  = np.tile(vector, (len(self.b), 1))
        self.output_history = np.tile(vector, (len(self.a), 1))
        self.old_output = np.copy(self.output_history[0])

    def output(self, idx=0): # idx: 0 - current, 1, previous, ...
        return self.output_history[idx]

    def speed(self):
        return self.output_history[0] - self.output_history[1]

    def new_point(self, vector):
        self.input_history = np.roll(self.input_history, 1, axis=0)
        self.old_output = np.copy(self.output_history[0])
        self.output_history = np.roll(self.output_history, 1, axis=0)
        self.input_history[0] = vector
        self.output_history[0] = (np.matmul(self.b.T, self.input_history) - np.matmul(self.a[1:].T, self.output_history[1:]))/self.a[0]
        return self.output()

    def skip_one(self):
        self.new_point(self.output())


class Car:
    def __init__(self, bbox, first=False):
        self.fps = 25.2 # 1261 frames in 50 secs
        self.nframe = 3 # num of frames to average
        self.threshold_found = 1
        self.bbox = bbox
        self.filtered_bbox = FilterAvg(bbox, 1/self.nframe*np.ones(self.nframe, dtype=np.float32), np.array([1.0, 0]))
        x,y = self.calculate_position(bbox)
        self.x = FilterAvg(x, 1/self.nframe*np.ones(self.nframe, dtype=np.float32), np.array([1.0, 0]))
        self.y = FilterAvg(y, 1/self.nframe*np.ones(self.nframe, dtype=np.float32), np.array([1.0, 0]))
        self.found = True
        self.num_lost = 0
        self.num_found = 0
        self.display = first
        

    def calculate_position(self, bbox):
        pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
        dst = cv2.perspectiveTransform(pos, transform_matrix).reshape(-1, 1)
        dx = (dst[0]-warped_size[0]/2.0)/pix_per_meter[0]
        dy = (warped_size[1]-dst[1])/pix_per_meter[1]
        return dx, dy


    def get_window(self):
        return self.filtered_bbox.output()

    def one_found(self):
        self.num_lost = 0
        if not self.display:
            self.num_found += 1
            if self.num_found > self.threshold_found:
                self.display = True

    def one_lost(self):
        self.num_found = 0
        self.num_lost += 1
        if self.num_lost > self.threshold_found:
            self.found = False

    def update_car(self, bboxes):
        current_window = self.filtered_bbox.output(0)
        intersection = np.zeros(4, dtype = np.float32)
        for idx, bbox in enumerate(bboxes):
            intersection[0:2] = np.maximum(current_window[0:2], bbox[0:2])
            intersection[2:4] = np.minimum(current_window[2:4], bbox[2:4])
            if (area(bbox)>0) and area(current_window) and ((area(intersection)/area(current_window)>0.8) or (area(intersection)/area(bbox)>0.8)):
                self.one_found()
                self.filtered_bbox.new_point(bbox)
                x,y=self.calculate_position(bbox)
                self.x.new_point(x)
                self.y.new_point(y)
                bboxes.pop(idx)
                return

        self.one_lost()
        self.filtered_bbox.skip_one()
        #self.position.skip_one()

    def draw(self, img, color=(0, 255, 0), thickness=2):
        if not self.display: return None, None
        window = self.filtered_bbox.output(0).astype(np.int32)
        rect = window # self.bbox # window
        rect_prev = self.filtered_bbox.output(1).astype(np.int32)
        # cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness)
        # cv2.rectangle(image,(x,y),( x + w, y + h ),(0,255,0),2)

        x,y = self.calculate_position(rect)
        vx = self.x.speed()*self.fps*3.6
        vy = self.y.speed()*self.fps*3.6
        label1 = '%.1f %.1fm' % (x, y) 
        label2 = '%.1f %.1fkm/h' % (vx, vy) 
        cv2.putText(img, label1, (int(rect[0]), int(rect[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2, color=color)
        cv2.putText(img, label2, (int(rect[0]), int(rect[3]+20)), # int(rect[1]+25)), #
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, thickness=2, color=color)
            
        label = label1 + ',  ' + label2
        return window, label

        

def area(bbox):
    return float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))




def get_processor(nbins=10):
    bins = nbins
    l_params = deque(maxlen=bins)
    r_params = deque(maxlen=bins)
    l_radius = deque(maxlen=bins)
    r_radius = deque(maxlen=bins)
    weights = np.arange(1,bins+1)/bins
    def process_image(image):
        global cars
        global first

        img, img_size = util_cal.get_undistorted_image(image, mtx, dist)
        src, dst = util_cal.get_transform_points(img_size)
        bird, M, Minv = util_cal.warp_image(img,src,dst,(img_size[1],img_size[0]))
        # util_cal.plt_n([new_img,img,bird],['original','undistorted','birds-eye'])
        #bird = util_pipe.histogram_equalize(bird)
        #img_bin = util_pipe.color_threshold(bird,s_thresh=(200,255),v_thresh=(200,255))
        #img_bin = util_pipe.pipeline_edge(bird,s_thresh=(150,255),g_thresh=(150,255))
        img_bin = util_pipe.pipeline_YW(bird)

        if slide_window_only or len(l_params)==0:  
            left_fit,right_fit,left_curverad,right_curverad,out_img = util_lane.slide_window_fit(img_bin) 
        else:
            left_fit,right_fit,left_curverad,right_curverad,out_img = util_lane.using_prev_fit(img_bin,   
                                                                np.average(l_params,0,weights[-len(l_params):]),
                                                                np.average(r_params,0,weights[-len(l_params):]))
        
        l_params.append(left_fit)
        r_params.append(right_fit)
        l_radius.append(left_curverad)
        r_radius.append(right_curverad)

        img_lane, new_lane, resized_bird = util_lane.mapping_fit_lane(bird, img_bin, img,  
                        np.average(l_params,0,weights[-len(l_params):]),
                        np.average(r_params,0,weights[-len(l_params):]),
                        np.average(l_radius,0,weights[-len(l_params):]),
                        np.average(r_radius,0,weights[-len(l_params):]), Minv)

        # resized_img_out = cv2.resize(img_out,(320,180))

        exec('from util_'+args.vdmodel+' import *')
        if args.vdmodel=='svm':
            img_out = np.copy(img)
        else:
            img_out = np.copy(img_lane)
        img_out, bboxes = eval('car_'+args.vdmodel)(img_out) # (img_lane)
        #img_out = eval('car_'+args.vdmodel)(img_out) 

        for car in cars:
            car.update_car(bboxes)

        for bbox in bboxes:
            cars.append(Car(bbox, first))

        tmp_cars = []
        for car in cars:
            if car.found:
                tmp_cars.append(car)
        cars = tmp_cars
        first = False

        #img_out = cv2.addWeighted(img_car, 0.7, new_lane, 0.3, 0)
        clips=[]
        labels=[]
        if len(cars)>0:
            for car in cars: 
                clip, label = car.draw(img_out)
                if clip is not None: clips.append(clip)
                if label is not None: labels.append(label)


        diagnostic_output = True
        if diagnostic_output == False: return img_out

        # put together multi-view output
        diag_img = np.zeros((720,1280,3), dtype=np.uint8)
        
        # original output (top left)
        diag_img[0:720,0:1280,:] = cv2.resize(img_out,(1280,720))
        
        resized_out_img = None
        if out_img is not None:
            if out_img.shape[0]>0 and out_img.shape[1]>0:
                resized_out_img = cv2.resize(out_img,(320,180)) # img_bin,(640,360))
        else:
            img_bin_stack = np.dstack((img_bin*255, img_bin*255, img_bin*255))
            resized_out_img = cv2.resize(img_bin_stack,(320,180))

        if resized_out_img is not None:
            diag_img[0:180,960:1280, :] = resized_out_img

        img_hist = util_pipe.histogram_equalize(img)
        img_hist_bird = util_lane.perspective_transform(img_hist, mtx_car)
        resized_img_hist = cv2.resize(img_hist,(320,180))
        resized_img_hist_bird = cv2.resize(img_hist_bird,(320,180))
        diag_img[0:180,640:960, :] = resized_img_hist
        diag_img[0:180,960:1280, :] = resized_img_hist_bird

        # resized_bird = cv2.resize(bird,(320,180))
        # #diag_img[0:180,320:640, :] = resized_bird

        # # plot the top-down color_warp as part of the result
        # warp_zero = np.zeros_like(img_bin).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # resized_color_warp = cv2.resize(color_warp,(320,180))
        # # resized_img_bin = cv2.resize(img_bin,(320,180))
        # cv2.addWeighted(resized_img_out, 0.2, resized_bird, 0.9, 0, resized_bird)
        
        # resized_bird = cv2.resize(bird,(320,180))
        diag_img[0:180,320:640, :] = resized_bird

        txt_img = diag_img[0:360,0:320, :]
        # cv2.putText(txt_img, 'Detected viehicles', (0,20), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2, cv2.LINE_AA)
        # for clip, label in zip(clips, labels):
        if len(clips) > 0:
            draw_thumbnails(txt_img, img_out, clips, labels)


        img_out = diag_img
        # img = np.copy(diag_img)

        return img_out

    return process_image


def process_image(img):
    exec('from util_'+args.vdmodel+' import *')
    image = eval('car_'+args.vdmodel)(img)
    
    return image


def do_image(img):
    return (get_processor(1))(img)

def do_image_file(fin, fout, save=True):
    img = mpimg.imread(fin)
    img_out = do_image(img)
    if save:
        mpimg.imsave(fout, img_out)
        print('Save output image to ',fout, ' ...')
    return img_out


def draw_thumbnails(img_cp, img, bboxes, labels, thumb_w=40, thumb_h=30, off_x=10, off_y=35):
    # cv2.putText(img_cp, 'Detected viehicles', (5,15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2) #, cv2.LINE_AA)
    x = off_x
    y = off_y
    # sort both bboxes and lables based on the size of area
    area = [(t[2]-t[0])*(t[3]-t[1]) for t in bboxes]
    _, bboxes = [list(x) for x in zip(*sorted(zip(area, bboxes), key=lambda pair: pair[0], reverse=True))]
    _, labels = [list(x) for x in zip(*sorted(zip(area, labels), key=lambda pair: pair[0], reverse=True))]
    
    for bbox, label in zip(bboxes, labels):
        thumbnail = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        vehicle_thumb = cv2.resize(thumbnail, (thumb_w, thumb_h))
        img_cp[int(y-thumb_h/2) : int(y-thumb_h/2 + thumb_h), x : x + thumb_w, :] = vehicle_thumb
        cv2.putText(img_cp, label, (x + thumb_w + off_x, y),
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1, color=(255,255,255))
        y += off_y


"""
Converts a string to y/n boolean
"""
def s2b(s):
	s = s.lower()
	return s == 'true' or s == 'yes' or s == 'y' or s == '1'

"""
parameters and defaults setting
"""
def set_args():
    parser = argparse.ArgumentParser(description='Vehicle Detection')
    parser.add_argument('-m', help='model-svm/yolo3/ssd/mrcnn', dest='vdmodel', type=str, default='mrcnn')
    parser.add_argument('-f', help='image file', dest='ifile', type=str, default=None)
    parser.add_argument('-v', help='video file', dest='vfile', type=str, default='project_video.mp4')
    parser.add_argument('-u', help='use lane info (Y/n)', dest='use_lane_info', type=s2b, default='y')
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    return args

   
def main():
    # test_output = 'test.mp4'
    # clip = VideoFileClip('test_video.mp4') # ('project_video.mp4')
    # test_clip = clip.fl_image(process_image)
    # test_clip.write_videofile(test_output, audio=False)

    if args.ifile is None:
        fname, fext = os.path.splitext(args.vfile)
        proj_output = fname + '_' + args.vdmodel + fext
        clip = VideoFileClip(args.vfile)
        if args.use_lane_info:
            proj_clip = clip.fl_image(get_processor(15))
        else:
            proj_clip = clip.fl_image(process_image)
        proj_clip.write_videofile(proj_output, audio=False)
    else:
        fin = args.ifile
        fname, fext = os.path.splitext(fin)
        fout = fname+'_out.jpg'
        do_image_file(fin, fout, save=True)

    # ch_output = 'challenge.mp4'
    # clip_ch = VideoFileClip('challenge_video.mp4')
    # challenge_clip = clip_ch.fl_image(process_image)
    # challenge_clip.write_videofile(ch_output, audio=False)

if __name__ == '__main__':
    heat_threshold = 1 # threshold of heatmap
    slide_window_only = False
    mtx, dist = util_cal.get_undistorted_params('calibration_pickle.p')
    mtx_car, minv_car, pixels_per_meter_car = util_cal.get_transform_car_params('projection_pickle.p')
    args = set_args()
    cars = []
    first= True
    main()
