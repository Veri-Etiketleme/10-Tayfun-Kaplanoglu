"""
Basic motion detection and tracking system

References: from pyimagesearch.com
    https://bit.ly/2C8edUi
    https://bit.ly/2HUIid2
"""
import argparse
import datetime
import json
import logging.config
import os
import pathlib
import sys
import time
# Third-party modules
import cv2
from imutils import resize
# import ipdb
# Own modules
from background_models.first_frame_model import FirstFrameModel
from background_models.weighted_average_model import WeightedAverageModel
from utilities.utils import get_full_command_line, setup_logging, timestamped, \
    unique_foldername, write_image
# Get the logger
logger = logging.getLogger('{}.{}'.format(os.path.basename(os.getcwd()),
                                          os.path.splitext(__file__)[0]))


if __name__ == '__main__':
    # configure root logger's level and format
    # NOTE: `logger` and `stdout_logger` will inherit config options from the
    # root logger
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c",
                    "--conf",
                    required=True,
                    help="path to the JSON configuration file")
    args = vars(ap.parse_args())

    # load the configuration file
    conf = json.load(open(args["conf"]))

    # =========================================================================
    #                   Processing configuration options
    # =========================================================================
    # Create 'main' directory for storing image results
    if conf["reports_dirpath"]:
        # TODO: first check that `reports_dirpath` exists. If it doesn't
        # exit create it.
        new_folder = os.path.join(conf["reports_dirpath"],
                                  timestamped("image_results"))
        new_folder = unique_foldername(new_folder)
        logger.debug("Creating folder {}".format(new_folder))
        pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
        conf["saved_folder"] = new_folder
        # Create folders for each set of images
        for fname in ["security_feed", "thresh", "frame_delta"]:
            if conf["save_{}_images".format(fname)]:
                image_folder = os.path.join(new_folder, fname)
                logger.debug("Creating folder {}".format(image_folder))
                pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)
            else:
                logger.debug("Folder for {} images not created".format(fname))
    else:
        logger.info("Images will not be saved")
        conf["saved_folder"] = None

    logger.info("Starting application")
    if conf["disable_logging"]:
        logger.info("Logging will be disabled")
    else:
        # IMPORTANT: logging is setup once main experiment directory is created
        # since we need the main directory to be ready for writing logging into it
        # Setup logging
        logger.debug("Setup logging")
        try:
            setup_logging(conf["logging_conf_path"], conf["saved_folder"])
        except (KeyError, OSError, ValueError) as e:
            logger.error(e)
            logger.warning("Logging couldn't be setup. The program will exit")
            sys.exit(1)
        else:
            logger.info("Logging was setup successfully!")

    # Validate background model
    models = [FirstFrameModel, WeightedAverageModel]
    model_names = [m.__background_model_name__ for m in models]
    background_model = None
    if conf["background_model"] not in model_names:
        logger.error("Background model ({}) is not supported".format(
                     conf["background_model"]))
        logger.error("Background models supported are {}".format(
                     models))
        logger.warning("Program will exit")
        sys.exit(1)
    else:
        logger.info("Background model used: {}".format(conf["background_model"]))
        for model in models:
            if model.__background_model_name__ == conf["background_model"]:
                background_model = model
                break
        assert background_model is not None

    # Validate gaussian kernel size
    ksize = conf["gaussian_kernel_size"]
    if not ksize["width"] % 2 or ksize["width"] <= 0:
        logger.error("Width of Gaussian kernel should be odd and positive")
        logger.warning("Program will exit")
        sys.exit(1)
    if not ksize["height"] % 2 or ksize["height"] <= 0:
        logger.error("Height of Gaussian kernel should be odd and positive")
        logger.warning("Program will exit")
        sys.exit(1)

    # Validate image format
    if conf["image_format"] not in ['jpg', 'jpeg', 'png']:
        logger.warning("Image format ({}) is not supported. png will be "
                       "used".format(conf["image_format"]))
        conf["image_format"] = 'png'

    if conf["resize_image_width"] == 0:
        logger.info("Images will not be resized")

    # Validate `start_frame`
    if conf["start_frame"] == 0 or not conf["start_frame"]:
        logger.warning("start_frame will be changed from {} to 1".format(
                       conf["start_frame"]))
        conf["start_frame"] = 1

    # Validate `end_frame`
    if conf["end_frame"] == 0 or not conf["end_frame"]:
        logger.info("end_frame is set to {}, thus motion detection will run "
                    "until last image".format(conf["end_frame"]))
        # TODO: use inf instead?
        conf["end_frame"] = 1000000

    # Setup camera: video file, list of images, or webcam feed
    logger.info("Setup camera")
    if conf["video_filepath"]:
        # Reading from a video file
        logger.info("Reading video file ...")
        camera = cv2.VideoCapture(conf["video_filepath"])
        logger.info("Finished reading video file")
    elif conf["image_dirpath"]:
        # Reading from a list of images with proper name format
        logger.info("Reading images ...")
        camera = cv2.VideoCapture(conf["image_dirpath"], cv2.CAP_IMAGES)
        logger.info("Finished reading images")
    else:
        # Reading from a webcam feed
        logger.info("Reading webcam feed ...")
        camera = cv2.VideoCapture(0)
        time.sleep(0.25)
        logger.info("Finished reading webcam feed")

    # Save configuration file and command line
    if conf["saved_folder"]:
        logger.info("Saving configuration file and command line")
        with open(os.path.join(conf["saved_folder"], 'conf.json'), 'w') as outfile:
            # ref.: https://stackoverflow.com/a/20776329
            json.dump(conf, outfile, indent=4, ensure_ascii=False)
        with open(os.path.join(conf["saved_folder"], 'command.txt'), 'w') as outfile:
            outfile.write(get_full_command_line())

    # ==========================================================================
    #                       Processing images/video
    # ==========================================================================
    logger.info("Start of images/video processing ...")

    # Initialize the first/average frame in the video file/webcam stream
    # NOTE 1: first frame can be used to model the background of the video stream
    # We assume that the first frame should not have motion, it should just
    # contain background
    # NOTE 2: the weighted mean of frames frame can also be used to model the
    # background of the video stream
    first_frame = True

    # Loop over the frames of the video
    # The first frame is the background image and is numbered as frame number 1
    frame_num = 2
    while True:
        logger.info("Processing frame #{}".format(frame_num))
        if conf["start_frame"] <= frame_num <= conf["end_frame"]:
            # Grab the current frame and initialize the occupied/unoccupied text
            # `grabbed` (bool): indicates if `frame` was successfully read from
            # the buffer
            (grabbed, frame) = camera.read()
            text = "Unoccupied"  # No activity in the room

            # If the frame could not be grabbed, then we have reached the end of
            # the video
            if not grabbed:
                logger.info("End of video")
                break

            # Preprocessing: prepare current frame for motion analysis
            # Resize the frame to 500 pixels wide, convert it to grayscale, and
            # blur it
            # NOTE: image width is used when image is resized. If width is 0,
            # image will not be resized.
            if conf["resize_image_width"] > 0:
                if frame.shape[1] <= conf["resize_image_width"]:
                    logger.debug("Image is being resized to a width ({}) that is "
                                 "greater than its actual width ({})".format(
                                  conf["resize_image_width"], frame.shape[1]))
                frame = resize(frame, width=conf["resize_image_width"])
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (ksize["width"], ksize["height"]), 0)

            # If frame representing background model is `None`, initialize it
            if first_frame:
                logger.debug("Starting background model ({})...".format(
                             conf["background_model"]))
                saving_cfg = {'saved_folder': conf['saved_folder'],
                              'image_format': conf['image_format']}
                background_model = background_model(gray, saving_cfg)
                first_frame = False
                continue

            # =================================================================
            #              Start of motion detection and tracking
            # =================================================================
            frameDelta = background_model.get_frame_delta(gray)

            # Threshold the delta image, dilate the thresholded image to fill
            # in holes, then find contours on thresholded image
            _, thresh = cv2.threshold(frameDelta,
                                      conf["delta_thresh"],
                                      255,
                                      cv2.THRESH_BINARY)
            thresh = cv2.dilate(thresh, None, iterations=2)
            (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)

            # Loop over the contours
            for c in cnts:
                # If the contour is too small, ignore it
                # `min-area`: minimum size (pixels) for a region of an image to be
                # considered actual “motion”
                if cv2.contourArea(c) < conf["min_area"]:
                    continue

                # Compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"

            # Draw the text (top left), timestamp (bottom left), and frame #
            # (top right) on the current frame
            # TODO: add as option the "Room Status" message
            # cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            if conf["show_datetime"]:
                datetime_now = datetime.datetime.now()
                # TODO: remove the following
                datetime_now = datetime_now.replace(hour=15)
                cv2.putText(frame,
                            datetime_now.strftime("%A %d %B %Y %I:%M:%S%p"),
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.35,
                            (0, 0, 255),
                            1)
            cv2.putText(frame,
                        "Frame # {}".format(frame_num),
                        (frame.shape[1] - 90, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 255),
                        1)

            # NOTE: path to the folder where three sets of images (security feed,
            # thresold and frame delta) will be saved
            if conf["saved_folder"]:
                image_sets = {'security_feed': frame,
                              'thresh': thresh,
                              'frame_delta': frameDelta}
                for iname, image in image_sets.items():
                    if conf["save_{}_images".format(iname)]:
                        inum = "{0:06d}".format(frame_num)
                        fname = "{}_{}.{}".format(iname,
                                                  inum,
                                                  conf["image_format"])
                        fname = os.path.join(conf["saved_folder"], iname ,fname)
                        write_image(fname, image)
                    else:
                        logger.debug("{} image not saved: frame # {}".format(
                                     iname, frame_num))

            # Check to see if the frames should be displayed to screen
            if conf["show_video"]:
                # Show the frame and record if the user presses a key
                cv2.imshow("Security Feed", frame)
                cv2.imshow("Thresh", thresh)
                cv2.imshow("Frame Delta", frameDelta)
                key = cv2.waitKey(1) & 0xFF

                # If the `q` key is pressed, break from the loop
                if key == ord("q"):
                    logger.info("Q key pressed. Quitting program ...")
                    break

        elif frame_num > conf["end_frame"]:
            logger.info("Reached end of frames: frame # {}".format(frame_num))
            break
        else:
            logger.info("Skipping frame number {}".format(frame_num))

        # Update frame number
        frame_num += 1

    logger.info("End of images/video processing")
    logger.info("Number of frames processed: {}".format(frame_num - 1))

    # Cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()

    logger.info("End of application")


