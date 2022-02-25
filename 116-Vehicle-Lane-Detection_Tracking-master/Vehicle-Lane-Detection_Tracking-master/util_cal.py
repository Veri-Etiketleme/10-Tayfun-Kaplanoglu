import numpy as np
import cv2
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# plot n cloumns of images
def plt_images(images, titles, columns):
    n = len(images)
    rows = n/columns
    if n%columns > 0: rows += 1
    plt.figure(figsize=(12,4))
    for i in range(n):
        plt.subplot(rows, columns, i+1, xticks=[], yticks=[])
        plt.imshow(images[i], cmap='gray')
        #plt.title(titles[i])
    plt.tight_layout() 
    plt.show()

# plot n images with title in a row    
def plt_n(images, titles):
    n = len(images)
    plt.figure(figsize=(14,4))
    for i, (image, title) in enumerate(zip(images, titles)):
        plt.subplot(1,n,i+1)
        plt.imshow(image, cmap='gray') 
        plt.axis('off')
        plt.title(title)
    plt.show()

# plot 2 views with marking src/dst points
def plt_views(image, warped, src=None, dst=None, title1="Undistorted", title2="Bird\'s eye view"):
    plt.figure(figsize=(14,4))
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    if src is not None:
        for i in range(4):
            plt.plot(src[i][0],src[i][1],'ro') #'rs')
    plt.title(title1)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(warped, cmap='gray')
    if dst is not None:
        for i in range(4):
            plt.plot(dst[i][0],dst[i][1],'ro') #'rs')
    
    plt.title(title2)
    plt.axis('off')



# calibrate and save params to pickle file
# nx- number of chessboard x grids
# ny- number of chessboard y grids
def calibrate_camera(images_path, save_path, nx=9, ny=6, plot=True):
    # prepare object points, like (0,0,0), (1,0,0), ....,(5,8,0)
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(images_path)
    image_size = None
    # Step through the list and search for chessboard corners
    imgs = []
    names= []
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if image_size == None:
            image_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if plot:
                if idx==11 or idx==13:
                    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    imgs.append(img)
                    names.append(fname)
    
    if plot:
        plt_n(imgs, names)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)
    # save the calibration parameters
    cal_para = {}
    cal_para["mtx"] = mtx
    cal_para["dist"] = dist
    pickle.dump(cal_para, open(save_path, "wb"))
    return mtx, dist

# Perspective transform matrix using vanish point
def get_transform_car_params(pickle_file):
    p = pickle.load(open(pickle_file,'rb'))
    mtx_car = p['perspective_transform']
    minv_car = p['inverse_transform']
    pixels_per_meter_car = p['pixels_per_meter']
    return mtx_car, minv_car, pixels_per_meter_car


# get undistorted images
# input - calibration file 
# output - undistored images
def get_undistorted_params(pickle_file):
    dist_pickle = pickle.load(open(pickle_file, 'rb'))
    mtx = dist_pickle['mtx']
    dist = dist_pickle['dist']
    return mtx, dist

# input: path_to_images, mtx, dist, plot_flag
# output: undistorted images
def get_undistorted_images_from_path(image_path, mtx, dist, plot=False):
    fnames = glob.glob(image_path) # './test_images/*.jpg')
    images=[]
    for fname in fnames:
        image = cv2.imread(fname)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

    imgs = get_undistorted_images(images, mtx, dist, fnames, plot)

    return imgs

# input: images, mtx, dist, image_names, plot_flag
# output: undistorted images
def get_undistorted_images(images, mtx, dist, names=None, plot=False):
    imgs = []
    for idx, img_in in enumerate(images):
        img = cv2.undistort(img_in, mtx, dist)
        img_size = np.shape(img)
        if img_size[2]==4:
            print('channel=4 -> ', names[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            print(np.shape(img))
        imgs.append(img)
        img_diff = np.sum((img-img_in)**2,axis=2)

        if plot:
            if names is not None: print(names[idx])
            plt_n([img_in, img, img_diff],['Original','Undistorted','Difference'])
    
    return imgs

#
def get_undistorted_image(image, mtx, dist):
    img = cv2.undistort(image, mtx, dist)
    img_size = np.shape(img)
    return img, img_size


# get an image and return the undistorted image   
def cal_undistort(img, objpoints, imgpoints):
    # Use cv2.calibrateCamera() and cv2.undistort()
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rot, trans = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    h, w = img.shape[:2]
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

# return RoI (Region of Interest) from the chessboard cornet point
def roi_from_corners(img, nx, ny):
    # Convert undistorted image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    img_size = (gray.shape[1], gray.shape[0])
    offset = 100
    # using the automatic detection of corners as source points
    # define 4 destination points dst
    src = np.float32([corners[0],corners[nx-1],corners[-1], corners[-nx]])
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset],
                     [img_size[0]-offset, img_size[1]-offset],
                     [offset, img_size[1]-offset]])
    return src, dst

# get an image and return the warped image
def warp_image(img, src, dst, img_size):
    # Apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return warped,M,Minv

# plot comparison of original, undistorted abd Bird's eye views
def plt_birds_view(img, mtx, dist, src, dst):
    undist = cv2.undistort(img, mtx, dist)
    img_size = (img.shape[1], img.shape[0])
    warped,_,_ = warp_image(undist, src, dst, img_size)
    plt_n([img, undist, warped], ['Original','Undistorted','Birds-eye'])


# set/get transform points 
# input - image size
# output- source and destination points - src, dst
def get_transform_points(img_size):
    src = np.float32([[581, 477],[699, 477],[896, 675],[384, 675]])
    dst = np.float32([[384,   0],[896,   0],[896, 720],[384, 720]])

    """
    src = np.float32([[581, 477],[699, 477],[903, 675],[377, 675]])
    dst = np.float32([[377,   0],[903,   0],[903, 720],[377, 720]])

    #=== 
    src = np.float32([[585,460],[203,720],[1127,720],[695,460]])
    dst = np.float32([[320,  0],[320,720],[ 960,720],[960,  0]])
    
    
    # img_size0 = 720
    # img_size1 = 1280
    offset = 150
    src = np.float32([(603, 445), (677, 445), 
           (1105, img_size[0]), (205, img_size[0])])
    dst = np.float32([(205 + offset, 0), (1105 - offset, 0), 
           (1105 - offset, img_size[0]), (205 + offset, img_size[0])])

    #  
    margin = .2 * img_size[1]
    top = np.uint(img_size[0]/1.5)
    btm = np.uint(img_size[0]*.95)
    center = np.uint(img_size[1]/2)
    top_l = center - .2*np.uint(img_size[1]/2)
    top_r = center + .2*np.uint(img_size[1]/2)
    btm_l = center - .9*np.uint(img_size[1]/2)
    btm_r = center + .9*np.uint(img_size[1]/2)
    
    src = np.float32([[btm_l,btm],[btm_r,btm],
                      [top_r,top],[top_l,top]])

    dst = np.float32([[margin,img_size[0]],[img_size[1]-margin,img_size[0]],
                      [img_size[1]-margin,0],[margin,0]])
    
     # ---
    src = np.float32(
    [[(img_size[0] / 2.) - 55., img_size[1] / 2. + 100.],
    [((img_size[0] / 6.) - 10.), img_size[1]],
    [(img_size[0] * 5. / 6.) + 60., img_size[1]],
    [(img_size[0] / 2. + 55.), img_size[1] / 2. + 100.]])
    dst = np.float32(
    [[(img_size[0] / 4.), 0],
    [(img_size[0] / 4.), img_size[1]],
    [(img_size[0] * 3. / 4.), img_size[1]],
    [(img_size[0] * 3. / 4.), 0]])
    """
    
    return src, dst

