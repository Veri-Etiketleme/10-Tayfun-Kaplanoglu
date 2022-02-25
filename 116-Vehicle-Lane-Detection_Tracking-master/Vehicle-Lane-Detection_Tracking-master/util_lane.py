import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import main
import util_pipe

# Define a class to keep track of lane line detection
# This information will help the main pipeline code to decide if the current polyfit is good or bad
class Line():
    def __init__(self):

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        # radius of curvature of this lane
        self.radius_of_curvature = None


left_line = Line()
right_line = Line()



# Implement Sliding Windows and Fit a Polynomial
# img - take a warped binary image as input
# nwindows - choose the number of sliding windows
# margin - set the width of the windows +/- margin
# minpix - set minimum number of pixels found to recenter window
def slide_window_fit(img, nwindows=9, margin=80, minpix=40):
    ict=0
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = (np.dstack((img, img, img)) * 255) #.astype(np.uint8)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Rectangle data for visualization
    ##rectangle_data = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        ##rectangle_data.append((win_y_low, win_y_high, win_xleft_low, win_xleft_high, win_xright_low, win_xright_high))
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # to plot
    #out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    #out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    if len(leftx)==0 or len(rightx)==0 :
        ict += 1
        print('search not found ', ict, ' times.')
        return left_line.current_fit,right_line.current_fit,left_line.radius_of_curvature,right_line.radius_of_curvature,out_img

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ##visual_data = (rectangle_data, histogram)
    ##return left_fit, right_fit, left_lane_inds, right_lane_inds, visual_data

    # Store in Line class
    left_line.current_fit = left_fit
    right_line.current_fit = right_fit
    
    # Generate x and y values for plotting
    try:
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        out_img[ploty.astype('int'),left_fitx.astype('int')] = [0, 255, 255]
        out_img[ploty.astype('int'),right_fitx.astype('int')] = [0, 255, 255]
    except:
        pass

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

    # Calculate radii of curvature in meters
    y_eval = np.max(ploty)  # Where radius of curvature is measured
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    # Store current radius of curvatures  
    left_line.radius_of_curvature = left_curverad  
    right_line.radius_of_curvature = right_curverad
    
    return left_fit, right_fit, left_curverad, right_curverad, out_img

#
# Skip the sliding windows step once knowing where the lines are
# left_fit  - previous good left_fit
# right_fit - previous good right fit
def using_prev_fit(img, left_fit, right_fit, margin=100):
    # Assume you now have a new warped binary image 
    # It's now much easier to find line pixels!
    # img - from the next frame of video (also called "binary_warped")
    
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img)) * 255

    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # to plot
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2) 
    except:
        return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
    
    else:
        # Check difference in fit coefficients between last and new fits  
        left_line.diffs = left_line.current_fit - left_fit
        right_line.diffs = right_line.current_fit - right_fit
        if (left_line.diffs[0]>0.001 or left_line.diffs[1]>0.4 or left_line.diffs[2]>150):
            return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None

        if (right_line.diffs[0]>0.001 or right_line.diffs[1]>0.4 or right_line.diffs[2]>150):
            return left_line.current_fit, right_line.current_fit, left_line.radius_of_curvature, right_line.radius_of_curvature, None
        
        
        # Store and replace the fit values in Line class
        left_line.current_fit = left_fit
        right_line.current_fit = right_fit


        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, deg=2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, deg=2)

        # Generate x and y values for plotting
        try:
            ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # Calculate radii of curvature in meters
            y_eval = np.max(ploty)  # Where radius of curvature is measured
            left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
            right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])     

            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            out_img[ploty.astype('int'),left_fitx.astype('int')] = [0, 255, 255]
            out_img[ploty.astype('int'),right_fitx.astype('int')] = [0, 255, 255]
        except:
            pass

        # Stash away the curvatures  
        left_line.radius_of_curvature = left_curverad  
        right_line.radius_of_curvature = right_curverad

        return left_fit, right_fit, left_curverad, right_curverad, out_img
    
    
#
# Generate x and y values for plotting
def get_fit_xy(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    if left_fit is None: 
        left_fitx is None
    else:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    if right_fit is None: 
        right_fitx is None
    else:
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    return left_fitx, right_fitx, ploty

#
# plot the fitting result
def view_slide_window_fit(binary, plot=False):
    left_fit, right_fit, left_curverad, right_curverad, out_img = slide_window_fit(binary)

    # Generate x and y values for plotting
    left_fitx, right_fitx, ploty = get_fit_xy(binary, left_fit, right_fit)
    
    if plot:
        plt.imshow(out_img)
        if left_fitx is not None:
            plt.plot(left_fitx, ploty, color='yellow')
        if right_fitx is not None:
            plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit, left_curverad, right_curverad, out_img


#
# plot fiting lane in original image
def plt_fit_lane(img, binary, src, dst, left_curverad, right_curverad):
    M = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    result = mapping_fit_lane(binary, img, left_fitx, right_fitx, left_curverad, right_curverad, M)
    fig = plt.figure(figsize=(16, 9))
    plt.imshow(result)
    plt.show()

# perspective transform through transform matrix M
def perspective_transform(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped



# Mapping the warped image back to original image
def mapping_fit_lane(bird, binary_img, origin_img, left_fit, right_fit, left_curverad, right_curverad, Minv):
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    left_fitx, right_fitx, ploty = get_fit_xy(binary_img, left_fit, right_fit)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    midpoint = np.int(origin_img.shape[1]/2)
    middle_of_lane = (right_fitx[-1] - left_fitx[-1]) / 2.0 + left_fitx[-1]
    offset = (midpoint - middle_of_lane) * xm_per_pix

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 0, 255)) # (0,255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
    cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(255,255,0), thickness=15) #color=(0,255,255)

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = perspective_transform(color_warp, Minv)
    # Combine the result with the original image
    result = cv2.addWeighted(origin_img, 0.7, newwarp, 0.3, 0)
    
    radius = np.mean([left_curverad, right_curverad])

    # Add radius and offset calculations to top of video
    # txt_img = np.copy(origin_img)
    # cv2.rectangle(txt_img,(0,0),(320,180),(0,0,0), -1)
    
    # cv2.putText(result,"L. Lane Radius: " + "{:0.2f}".format(left_curverad/1000) + 'km', org=(30,50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=0.6, color=(255,255,255), lineType = cv2.LINE_AA, thickness=1)
    # cv2.putText(result,"R. Lane Radius: " + "{:0.2f}".format(right_curverad/1000) + 'km', org=(30,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=0.6, color=(255,255,255), lineType = cv2.LINE_AA, thickness=1)
    # cv2.putText(result,"C. Position: " + "{:0.2f}".format(offset) + 'm', org=(30,150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #         fontScale=0.6, color=(255,255,255), lineType = cv2.LINE_AA, thickness=1)

    # cv2.addWeighted(txt_img, 0.3, result, 0.7, 0, result)

    # img_bin = np.dstack((binary_img*255, binary_img*255, binary_img*255))
    # resized_out_img = cv2.resize(img_bin,(320,180))
    # result[0:180,960:1280, :] = resized_out_img

    # histogram undistorted view (top middle)
    # img_hist = util_pipe.histogram_equalize(origin_img)
    # img_hist_bird = perspective_transform(img_hist, mtx_car)
    #resized_img_hist = cv2.resize(img_hist,(320,180))
    #resized_img_hist_bird = cv2.resize(img_hist_bird,(320,180))
    #result[0:180,640:960, :] = resized_img_hist
    #result[0:180,960:1280, :] = resized_img_hist_bird
    
    # # plot the bird's eye view
    # # binary = np.dstack((binary_img*255, binary_img*255, binary_img*255))
    resized_bird = cv2.resize(bird,(320,180))
    # result[0:180,320:640, :] = resized_bird

    # # plot the top-down color_warp as part of the result
    resized_color_warp = cv2.resize(color_warp,(320,180))
    cv2.addWeighted(resized_color_warp, 0.2, resized_bird, 0.9, 0, resized_bird)

    # result[0:180,320:640, :] = resized_bird
    # plt.imshow(result)
    return result, newwarp, resized_bird





