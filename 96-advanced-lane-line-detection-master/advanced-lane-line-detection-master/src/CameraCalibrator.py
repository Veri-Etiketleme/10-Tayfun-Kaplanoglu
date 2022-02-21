"""Camera Calibrator"""
import cv2
import pickle
import numpy as np


class CameraCalibrator():
    """Camera Calibrator"""

    def __init__(self):
        """Initialize camera calibrator"""
        self.dist_coefficients = None
        self.undistort_matrix = None

    def calibrate(self, calibration_images, chessboard_shape):
        """Process images and return calibration coefficients and matrix"""
        assert isinstance(calibration_images, list)
        assert isinstance(chessboard_shape, tuple)

        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_shape[0] * chessboard_shape[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_shape[0], 0:chessboard_shape[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        for filename in calibration_images:
            img = cv2.imread(filename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, chessboard_shape, None)

            if ret is True:
                objpoints.append(objp)
                imgpoints.append(corners)

        img_size = (1280, 720)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

        self.dist_coefficients = dist
        self.undistort_matrix = mtx

    def undistort(self, img):
        """Undistort given image"""
        assert self.dist_coefficients is not None
        assert self.undistort_matrix is not None

        return cv2.undistort(img, self.undistort_matrix, self.dist_coefficients, None, self.undistort_matrix)

    def saveParameters(self, location):
        """Save calibration parameters into pickle file"""
        dist_pickle = {}
        dist_pickle["mtx"] = self.undistort_matrix
        dist_pickle["dist"] = self.dist_coefficients
        pickle.dump(dist_pickle, open(location, "wb"))

    def loadParameters(self, location):
        """Load calibration parameters from location"""
        dist_pickle = pickle.load(open(location, "rb"))
        self.undistort_matrix = dist_pickle["mtx"]
        self.dist_coefficients = dist_pickle["dist"]
