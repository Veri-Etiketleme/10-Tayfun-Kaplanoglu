"""Define a class to track the lane"""
import cv2
import numpy as np

from Line import Line


class Lane():
    """Represents a road lane"""

    def __init__(self, image_size):
        """Constuct the object."""
        self.left_line = Line(frame_memory=7)
        self.right_line = Line(frame_memory=7)

        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension

        self.polynom_search_margin = 100

        self.image_size = image_size  # Size of input images, usually (720, 1280)

        self.center_polynom = None
        self.center_curvature = 0.0
        self.center_offset = 0.0

        self.undetected_frame_count = 0
        self.previos_detection = False

    def reset(self):
        """Reset lane to its initial state"""
        self.left_line.reset()
        self.right_line.reset()
        self.center_polynom = None
        self.center_curvature = 0.0
        self.center_offset = 0.0
        self.undetected_frame_count = 0

    def detect_lane(self, image):
        """Detect lane lines by applying a sliding window."""
        leftx = lefty = rightx = righty = []

        # If lines have been detected previously try to find current ones along polynomial
        if self.left_line.detected and self.right_line.detected:
            leftx, lefty, rightx, righty = self.do_margin_based_search(image)
            self.check_detected_lines(leftx, lefty, rightx, righty)

        # If lines havent been detected yet try to find them by sliding window
        if not self.left_line.detected and not self.right_line.detected:
            leftx, lefty, rightx, righty = self.do_sliding_window_search(image)
            self.check_detected_lines(leftx, lefty, rightx, righty)

        # If lines are not detected for a row of frames reset previous detections
        if not self.left_line.detected and not self.right_line.detected:
            self.undetected_frame_count += 1
            if (self.undetected_frame_count > 14):
                self.reset()
        else:
            self.undetected_frame_count = 0

        # Update line information
        if self.left_line.detected:
            self.left_line.update(x=lefty, y=leftx)
            self.previos_detection = True

        if self.right_line.detected:
            self.right_line.update(x=righty, y=rightx)

        # Update lane information based on center polynom
        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            self.center_polynom = (self.left_line.best_fit + self.right_line.best_fit) / 2

            self.caluculate_lane_curvature()
            self.Calculate_center_offset()

    def check_detected_lines(self, leftx, lefty, rightx, righty):
        """Check if the detected lane line pixels are plausible"""
        is_valid_input = len(leftx) > 3 and len(rightx) > 3
        lines_are_plausible = False
        lines_are_parallel = False

        if is_valid_input:
            detected_left_line = Line(x=lefty, y=leftx)
            detected_right_line = Line(x=righty, y=rightx)

            # Check if they are parallel
            first_coefficients_diff = np.abs(detected_left_line.current_fit[2] - detected_right_line.current_fit[2])
            second_coefficients_diff = np.abs(detected_left_line.current_fit[1] - detected_right_line.current_fit[1])

            lines_are_parallel = first_coefficients_diff < 0.0005 and second_coefficients_diff < 0.55

            # Check if the lines have plausible distance
            distance = np.abs(detected_left_line.current_fit(719) - detected_right_line.current_fit(719))
            # print('Distance: ' + str(distance))
            lines_are_plausible = 380 < distance < 550

        detection_ok = is_valid_input & lines_are_plausible & lines_are_parallel

        self.left_line.detected = detection_ok
        self.right_line.detected = detection_ok

    def find_line_centers(self, image):
        """Find a left and right lane center position by histogram"""
        histogram = np.sum(image[int(self.image_size[0] / 2):, :], axis=0)
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def do_margin_based_search(self, image):
        """Do line search along the latest polynomial fit"""
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.polynom_search_margin

        left_lane_inds = ((nonzerox > (self.left_line.best_fit[2] * (nonzeroy**2) + self.left_line.best_fit[1] * nonzeroy + self.left_line.best_fit[0] - margin)) &
                          (nonzerox < (self.left_line.best_fit[2] * (nonzeroy**2) + self.left_line.best_fit[1] * nonzeroy + self.left_line.best_fit[0] + margin)))

        right_lane_inds = ((nonzerox > (self.right_line.best_fit[2] * (nonzeroy**2) + self.right_line.best_fit[1] * nonzeroy + self.right_line.best_fit[0] - margin)) &
                           (nonzerox < (self.right_line.best_fit[2] * (nonzeroy**2) + self.right_line.best_fit[1] * nonzeroy + self.right_line.best_fit[0] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def do_sliding_window_search(self, image):
        """Do sliding window line search based on histogram peaks"""
        nwindows = 9
        window_height = np.int(self.image_size[0] / nwindows)
        half_window_width = 100
        minpx_to_recenter = 50

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set window search starting positions
        leftx_current, rightx_current = self.find_line_centers(image)

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.image_size[0] - (window + 1) * window_height
            win_y_high = self.image_size[0] - window * window_height
            win_xleft_low = leftx_current - half_window_width
            win_xleft_high = leftx_current + half_window_width
            win_xright_low = rightx_current - half_window_width
            win_xright_high = rightx_current + half_window_width

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpx_to_recenter:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpx_to_recenter:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def caluculate_lane_curvature(self):
        """Calculate lane curvature of left and right lane"""
        assert self.center_polynom is not None

        y = np.array(np.linspace(0, 719, num=10))
        x = np.array([self.center_polynom(x) for x in y])
        y_eval = np.max(y)

        # Calculate the new radii of curvature
        world_space_fit = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
        self.center_curvature = ((1 + (2 * world_space_fit[0] * y_eval / 2. + world_space_fit[1]) ** 2) ** 1.5) / np.absolute(2 * world_space_fit[0])

    def Calculate_center_offset(self):
        """Calculate vehicle center offset"""
        assert self.center_polynom is not None

        lane_center = self.center_polynom(self.image_size[0] - 1)
        vehicle_center = self.image_size[1] / 2

        self.center_offset = (vehicle_center - lane_center) * self.xm_per_pix

    def get_debug_image(self, image):
        """Update debug image"""
        out_img = np.dstack((image, image, image)) * 255

        if self.left_line.best_fit is not None and self.right_line.best_fit is not None:
            out_img[self.left_line.allx, self.left_line.ally] = [255, 0, 0]
            out_img[self.right_line.allx, self.right_line.ally] = [0, 0, 255]

            left_fit = self.left_line.best_fit.c
            right_fit = self.right_line.best_fit.c

            # Generate x and y values for plotting
            ploty = np.linspace(0, image.shape[0] - 1, image.shape[0])
            left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

            left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - self.polynom_search_margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + self.polynom_search_margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))

            right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - self.polynom_search_margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + self.polynom_search_margin, ploty])))])
            right_line_pts = np.hstack((right_line_window1, right_line_window2))

            window_img = np.zeros_like(out_img)
            cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))

            out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return out_img
