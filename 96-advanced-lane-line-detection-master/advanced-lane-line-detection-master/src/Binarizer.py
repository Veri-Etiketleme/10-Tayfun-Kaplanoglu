"""Binarizer"""
import numpy as np
import cv2


class Binarizer():
    """Binarizer - processes an image"""

    def __init__(self):
        """Initialize member variables"""
        self.sobelx = None
        self.sobely = None

    def process(self, image):
        """Process image and output a binarized version"""
        yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        Y = yuv[:, :, 0]
        V = yuv[:, :, 2]

        Y_binary = np.zeros_like(Y)
        Y_binary[(Y >= 200) & (Y <= 255)] = 1

        V_binary = np.zeros_like(V)
        V_binary[(V >= 0) & (V <= 100)] = 1

        self.sobelx = cv2.Sobel(Y, cv2.CV_64F, 1, 0, ksize=5)
        self.sobely = cv2.Sobel(Y, cv2.CV_64F, 0, 1, ksize=5)

        gradx = self.abs_sobel_threshold(orient='x', thresh=(20, 100))
        grady = self.abs_sobel_threshold(orient='y', thresh=(20, 100))

        combined = np.zeros_like(Y)
        combined[(Y_binary == 1) | (V_binary == 1)] = 1

        return combined

    def abs_sobel_threshold(self, orient='x', thresh=(0, 255)):
        """Apply sobel filter and thresholding"""
        if orient == 'x':
            abs_sobel = np.absolute(self.sobelx)
        if orient == 'y':
            abs_sobel = np.absolute(self.sobely)

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary
