"""Process images based on the defined pipeline"""
# import cv2
# import matplotlib.pyplot as plt

from Binarizer import Binarizer
from CameraCalibrator import CameraCalibrator
from Lane import Lane
from moviepy.editor import VideoFileClip
from optparse import OptionParser
from Visualizer import Visualizer
from Warper import Warper


# =============================================================================
# Get command line arguments
# =============================================================================
parser = OptionParser()
(options, args) = parser.parse_args()

input_video_name = args[0]
output_video_name = 'result_' + input_video_name

# =============================================================================
# Create processing instances
# =============================================================================
calibrator = CameraCalibrator()
binarizer = Binarizer()
warper = Warper()
visualizer = Visualizer(warper)
lane = Lane(image_size=(720, 1280))

calibrator.loadParameters("./camera_cal/calibration_coefficients.p")


# =============================================================================
# Preprocessing pipeline
# =============================================================================
def process_image(image):
    """Process a single image"""
    undist = calibrator.undistort(image)
    binarized = binarizer.process(undist)
    warped = warper.warp(binarized)

    lane.detect_lane(warped)

    debug_image = lane.get_debug_image(warped)

    visualizer.draw_debugging_output(undist, binarized, warped, debug_image)
    visualizer.draw_text_info(undist, lane.center_curvature, lane.center_offset)
    result = visualizer.draw_lane_on_road(undist, lane)

    return result


# image = cv2.imread('./test_images/straight_lines2.jpg')
# output = process_image(image)
# plt.imshow(output[..., ::-1])
# plt.show()

# =============================================================================
# Process video file
# =============================================================================
clip1 = VideoFileClip(input_video_name)
project_clip = clip1.fl_image(process_image)
project_clip.write_videofile(output_video_name, audio=False)
