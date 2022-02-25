import numpy as np
import cv2

def abs_sobel_thresh(img, sobel_kernel=3, orient='x', thresh=(0,255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient=='x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient=='y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude 
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_mag = np.uint8(255*mag/np.max(mag))
    binary_output = np.zeros_like(scaled_mag)
    binary_output[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Apply the following steps to img
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    # Create a binary mask where direction thresholds are me
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    # Return the binary image
    return binary_output

def histogram_equalize(img):
    r, g, b = cv2.split(img)
    red = cv2.equalizeHist(r)
    green = cv2.equalizeHist(g)
    blue = cv2.equalizeHist(b)
    return cv2.merge((red, green, blue))


# Gradient and Color Thresholds

def scale(img, factor=255.0):
    scale_factor = np.max(img)/factor
    return (img/scale_factor).astype(np.uint8)

def derivative(img, sobel_kernel=3):
    derivx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    derivy = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    gradmag = np.sqrt(derivx**2 + derivy**2)
    absgraddir = np.arctan2(derivy, derivx)
    return scale(derivx), scale(derivy), scale(gradmag), absgraddir

def grad(img, k1=3, k2=15):
    _,_,g,_ = derivative(img, sobel_kernel=k1)
    _,_,_,p = derivative(img, sobel_kernel=k2)
    return g,p

# lambda function of binary operation 
land = lambda *x: np.logical_and.reduce(x)
lor = lambda *x: np.logical_or.reduce(x)

# return image in threshold (min, max)
def threshold(img, thresh=(0,255)):
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return binary_output

# Test code -
# color = 'hls'
# c0,c1,c2=color_select(img, color)
# util_cal.plt_n([c0,c1,c2],[color[0],color[1],color[2]])
def color_select(img, color):
    color = color.upper()
    if color == 'RGB':
        img2 = img
    else:
        img2 = cv2.cvtColor(img, eval('cv2.COLOR_RGB2'+color))
    c0 = img2[:,:,0]
    c1 = img2[:,:,1]
    c2 = img2[:,:,2]
    return c0,c1,c2

# highlight effect
def highlight(img):
    r,g,b = color_select(img, 'rgb') 
    h,l,s = color_select(img, 'hls')
    h0 = threshold(r, (200, 255))
    h1 = threshold(g, (200, 255))
    h2 = threshold(s, (200, 255))
    return scale(lor(land(h0,h1),h2))

def pipeline_hbs(img, thresh=(200,255)):
    h=highlight(img)
    c=color_threshold(img, thresh)
    #h=threshold(h, thresh)
    return scale(lor(h,c))

def rgb_select(img, channel='R', thresh=(200, 255)):
    if channel == 'G':
        X = img[:, :, 1]
    elif channel == 'B':
        X = img[:, :, 2]
    else: # default to R
        X = img[:, :, 0]
    binary = np.zeros_like(X)
    binary[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary

def lab_select(img, channel='B', thresh=(190, 255)):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    if channel == 'L':
        X = img[:, :, 0]
    elif channel == 'A':
        X = img[:, :, 1]
    else: # default to B
        X = img[:, :, 2]
    binary = np.zeros_like(X)
    binary[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary

# Define a function that thresholds the channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, channel='S', thresh=(220, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    if channel == 'H':
        X = hls[:, :, 0]
    elif channel == 'L':
        X = hls[:, :, 1]
    else: # default to S
        X = hls[:, :, 2]
        
    binary = np.zeros_like(X)
    binary[(X > thresh[0]) & (X <= thresh[1])] = 1
    return binary

def color_mask(hsv,low,high):
    # Return mask from HSV
    mask = cv2.inRange(hsv, low, high)
    return mask

def apply_color_mask(hsv,img,low,high):
    # Apply color mask to image
    mask = cv2.inRange(hsv, low, high)
    res = cv2.bitwise_and(img,img, mask= mask)
    return res

def apply_yw_mask(img):
    image_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    yellow_hsv_low  = np.array([ 0,  100,  100])
    yellow_hsv_high = np.array([ 80, 255, 255])
    white_hsv_low  = np.array([ 0,   0,   160])
    white_hsv_high = np.array([ 255,  80, 255])
    mask_yellow = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = color_mask(image_HSV,white_hsv_low,white_hsv_high)
    mask_YW_image = cv2.bitwise_or(mask_yellow,mask_white)
    return mask_YW_image

def apply_yw_mask2(img):
    image_HSV = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    yellow_hsv_low  = np.array([ 50,  50,  50])
    yellow_hsv_high = np.array([ 110, 255, 255])
    white_hsv_low  = np.array([ 200,  200, 200])
    white_hsv_high = np.array([ 255,  255, 255])
    mask_yellow = color_mask(image_HSV,yellow_hsv_low,yellow_hsv_high)
    mask_white = color_mask(image_HSV,white_hsv_low,white_hsv_high)
    mask_YW_image = scale(lor(mask_yellow,mask_white))
    return mask_YW_image

def color_threshold(img, s_thresh=(0,255), v_thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1   
    
    # r_binary = rgb_select(img, channel='R', thresh=r_thresh)

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output
    

# Return the binary Thresholded image by combining multiple binary thresholds
def combine(img, l_thresh=(215,255), b_thresh=(145,255)):
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]  # Detect white lines
    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]  # Detect yellow lines
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

    combined_binary = np.zeros_like(b_binary)
    combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

    return combined_binary, l_binary, b_binary

# pipeline of all binaries
def pipeline_grad(img, x_thresh=(20, 255), r_thresh=(220,255), s_thresh=(100, 255)):
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=x_thresh)
    r_binary = rgb_select(img, channel='R', thresh=r_thresh)
    s_binary = hls_select(img, channel='S', thresh=s_thresh)
    l_binary = hls_select(img, channel='L', thresh=s_thresh)
    yw_binary = apply_yw_mask(img)
    yw_binary[(yw_binary !=0)] = 1
    # c_binary,_,_ = combine(img)
    combined = np.zeros_like(gradx)
    combined[(yw_binary == 1) & (r_binary == 1) | (s_binary == 1) ] = 1 
    # combined[((l_binary == 1) & (s_binary == 1) | (gradx == 1) | (yw_binary == 1))] = 1
    # combined[((l_binary == 1) & (yw_binary == 1) | (gradx == 1) | (s_binary == 1) | (r_binary == 1))] = 1
    return combined

def pipeline_rsv(img, r_thresh=(180,255), s_thresh=(100,255), v_thresh=(100,255)):
    r_binary = rgb_select(img, channel='R', thresh=r_thresh)
    c_binary = color_threshold(img, s_thresh, v_thresh)
    output = np.zeros_like(r_binary)
    output[(r_binary == 1) | (c_binary == 1)] = 1
    return output

def pipeline_lb(img, l_thresh=(220,255), b_thresh=(190,255)):
    # HLS L-channel Threshold (using default parameters)
    l_binary = hls_select(img, channel='L', thresh=l_thresh)
    # Lab B-channel Threshold (using default parameters)
    b_binary = lab_select(img, channel='B', thresh=b_thresh)
    # Combine HLS and Lab B channel thresholds
    combined = np.zeros_like(l_binary)
    combined[(l_binary == 1) | (b_binary == 1)] = 1
    return combined

def pipeline_rlb(img, r_thresh=(220,255), l_thresh=(215,255), b_thresh=(145,255)):
    r_binary = rgb_select(img, channel='R', thresh=r_thresh)
    c_binary = combine(img, l_thresh, b_thresh)
    output = np.zeros_like(r_binary)
    output[(r_binary == 1) | (c_binary == 1)] = 1
    return output

def pipeline_edge(img, s_thresh=(150, 255), g_thresh=(130,255)):
    gray = (0.5*img[:,:,0] + 0.4*img[:,:,1] + 0.1*img[:,:,2]).astype(np.uint8)
    g_binary = np.zeros_like(gray)
    g_binary[(gray >= g_thresh[0]) & (gray <= g_thresh[1])] = 1

    # switch to gray image for laplacian if 's' doesn't give enough details
    total_px = img.shape[0]*img.shape[1]
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
    mask_one = (laplacian < 0.15*np.min(laplacian)).astype(np.uint8)
    if cv2.countNonZero(mask_one)/total_px < 0.01:
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=21)
        mask_one = (laplacian < 0.075*np.min(laplacian)).astype(np.uint8)

    s_binary = hls_select(img, channel='S', thresh=s_thresh)
    mask_two = s_binary

    combined = np.zeros_like(g_binary)
    combined[((g_binary == 1) & ((mask_one == 1) | (mask_two == 1)) )] = 1

    return combined

# Filter the image, showing only a range of white and yellow
def pipeline_YW(image):
    # Filter White
    threshold = 200 
    high_threshold = np.array([255, 255, 255]) #Bright white
    low_threshold = np.array([threshold, threshold, threshold]) #Soft White
    mask = cv2.inRange(image, low_threshold, high_threshold)
    white_img = cv2.bitwise_and(image, image, mask=mask)

    # Filter Yellow
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #Changing Color-space, HSV is better for object detection
    #For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 
    high_threshold = np.array([110,255,255]) #Bright Yellow
    low_threshold = np.array([50,50,50]) #Soft Yellow   
    mask = cv2.inRange(hsv_img, low_threshold, high_threshold)
    yellow_img = cv2.bitwise_and(image, image, mask=mask)

    # Combine the two above images
    combined = cv2.addWeighted(white_img, 1., yellow_img, 1., 0.)
    gray = cv2.cvtColor(combined, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > 0)] = 1
    return binary



    