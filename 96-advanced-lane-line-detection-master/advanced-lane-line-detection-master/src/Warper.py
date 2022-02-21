"""Warper"""
import cv2
import numpy as np


class Warper():
    """Image Warper"""

    def __init__(self):
        """Initialize image warper"""
        src_top_left = [590, 450]
        src_top_right = [685, 450]
        src_bottom_right = [1120, 720]
        src_bottom_left = [190, 720]

        dest_top_left = [src_bottom_left[0] + (src_top_left[0] - src_bottom_left[0]) / 2, 0]
        dest_top_right = [src_top_right[0] + (src_bottom_right[0] - src_top_right[0]) / 2, 0]
        dest_bottom_right = [src_top_right[0] + (src_bottom_right[0] - src_top_right[0]) / 2, 720]
        dest_bottom_left = [src_bottom_left[0] + (src_top_left[0] - src_bottom_left[0]) / 2, 720]

        # dest_top_left = [src_bottom_left[0] + 300, 0]
        # dest_top_right = [src_top_right[0] + 300, 0]
        # dest_bottom_right = [src_top_right[0] + 300, 720]
        # dest_bottom_left = [src_bottom_left[0] + 300, 720]
        # dest_top_left = [200, 0]
        # dest_top_right = [1200, 0]
        # dest_bottom_right = [1200, 720]
        # dest_bottom_left = [200, 720]

        src_warp = np.float32([src_top_left, src_top_right, src_bottom_right, src_bottom_left])
        dest_warp = np.float32([dest_top_left, dest_top_right, dest_bottom_right, dest_bottom_left])

        self.warp_matrix = cv2.getPerspectiveTransform(src_warp, dest_warp)
        self.warp_matrix_inverse = cv2.getPerspectiveTransform(dest_warp, src_warp)

    def warp(self, image):
        """Warp an image based on matrix defined within the constructor"""
        image_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.warp_matrix, image_size, flags=cv2.INTER_LINEAR)

    def unWarp(self, image):
        """Warp an image based on matrix defined within the constructor"""
        image_size = (image.shape[1], image.shape[0])
        return cv2.warpPerspective(image, self.warp_matrix_inverse, image_size, flags=cv2.INTER_LINEAR)
