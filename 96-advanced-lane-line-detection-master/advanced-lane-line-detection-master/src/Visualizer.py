"""Visualizer draws information on a given image"""
import cv2
import numpy as np


class Visualizer():
    """Draws various information into an image"""

    def __init__(self, warper):
        """Construct the object."""
        self.warper = warper

    def draw_debugging_output(self, original_image, binary, warped_binary, warped_lane):
        """Draw debugging output into given image"""
        mini_width = original_image.shape[1] // 4
        mini_height = original_image.shape[0] // 4

        # Draw black debugging area
        cv2.rectangle(original_image, (0, 0), (original_image.shape[1], mini_height), (0, 0, 0), -1)

        # Draw a mini version of the binarized & thresholded image
        binary_mini = cv2.resize(binary, (mini_width, mini_height), None, 0, 0, cv2.INTER_LINEAR)
        binary_mini = binary_mini * 255
        binary_mini = np.dstack((binary_mini, binary_mini, binary_mini))
        drawx = original_image.shape[1] - mini_width
        original_image[0:0 + mini_height, drawx:drawx + mini_width] = binary_mini

        # Draw warped biniarized image with detected lane area
        lane_mini = cv2.resize(warped_lane, (mini_width, mini_height), None, 0, 0, cv2.INTER_LINEAR)
        drawx = drawx - mini_width - 30
        original_image[0:0 + mini_height, drawx:drawx + mini_width] = lane_mini

        return original_image

    def draw_text_info(self, image, radius, offset):
        """Draw lane information"""
        lane_curvature_text = 'Lane radius: ' + str(round(radius, 1)) + ' m'
        vehicle_distance_text = 'Distance to lane center: ' + str(round(offset, 1)) + ' m'
        cv2.putText(image, lane_curvature_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, vehicle_distance_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def draw_lane_on_road(self, original_image, lane):
        """Draw lane"""
        if lane.left_line.best_fit is not None and lane.right_line.best_fit is not None:
            left_fit = lane.left_line.best_fit.c
            right_fit = lane.right_line.best_fit.c

            # Generate x and y values for plotting
            ploty = np.linspace(0, original_image.shape[0] - 1, original_image.shape[0])
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            # Recast the x and y points into usable format for cv2.fillPoly()
            margin = 20
            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((left_line_window2, right_line_window1))

            mask = np.zeros_like(original_image)
            cv2.fillPoly(mask, np.int_([pts]), (0, 255, 0))
            cv2.fillPoly(mask, np.int_([left_line_pts]), (255, 0, 0))
            cv2.fillPoly(mask, np.int_([right_line_pts]), (0, 0, 255))
            mask = self.warper.unWarp(mask)

            # Draw the lane onto the warped blank image
            original_image = cv2.addWeighted(original_image, 1, mask, 0.3, 0)

        return original_image
