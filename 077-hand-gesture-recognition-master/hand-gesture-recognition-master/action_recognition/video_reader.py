import os
import numpy as np

import cv2
from PIL import Image
import re


class VideoReader:
    """This class reads specified frame numbers from video"""

    def read(self, video_path, frame_indices):
        raise NotImplementedError

    def __call__(self, video_path, frame_indices):
        return self.read(video_path, frame_indices)


class ImageDirReader(VideoReader):
    """Reads clip from directory with frames with a given file name pattern
    The image file names should contain one number (its frame index) and its
    extension must be in the list of supported extensions. Examples of valid file names:
        * frame_00001.jpg
        * image1.png
        * 0001.png
        * 1.tiff
    """

    def __init__(self, read_image_fn, supported_extensions=['.jpg', '.png', '.tiff']):
        self.read_image_fn = read_image_fn
        self.supported_extensions = supported_extensions

    def read(self, video_path, frame_indices):
        video = []
        image_filenames = {}

        # get files in dir that contain a supported extension and one number that represents
        # the frame index
        for filename in os.listdir(video_path):
            ext = os.path.splitext(filename)[1]
            frame_index_candidates = re.findall('\d+', filename)
            if ext in self.supported_extensions and len(frame_index_candidates) == 1:
                frame_index = int(frame_index_candidates[0])
                image_filenames[frame_index] = os.path.join(video_path, filename)

        # read images that have a frame index listed in frame_indices
        for frame_index in frame_indices:
            try:
                video.append(self.read_image_fn(image_filenames[frame_index]))
            except KeyError:
                raise RuntimeError("Requested frame with frame index {}, ".format(frame_index) +
                                   "does not exist is video path {}".format(video_path))
        return video


class VideoFileReader(VideoReader):
    """Reads clip from video file."""

    def read(self, video_path, frame_indices):
        video = []
        cap = cv2.VideoCapture(video_path)

        unique_indices = sorted(set(frame_indices))
        current_frame_idx = 0
        frame_map = {}
        for next_frame_idx in unique_indices:
            while current_frame_idx != next_frame_idx:
                status, frame = cap.read()
                current_frame_idx += 1
                if not status:
                    cap.release()
                    return video
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_map[next_frame_idx] = frame
        cap.release()
        video = [frame_map[i] for i in frame_indices]
        return video


def pil_read_image(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
            return np.asarray(img)


def accimage_read_image(path):
    try:
        import accimage
        return np.asarray(accimage.Image(path))
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_read_image(path)


def opencv_read_image(path):
    # read raw image, preserving its bit depth and number of channels
    image = cv2.imread(path, -1)

    # add channel dimension if not present (grayscale images for instance)
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    # interpret thermal images as signed ints since values can be negative
    if image.dtype.type == np.uint16:
        image = image.astype(np.int16)

    # convert BGR to RGB for color images
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def make_video_reader(video_format='frames', image_reader='opencv'):
    if image_reader == 'opencv':
        image_read_fn = opencv_read_image
    elif image_reader == 'pil':
        image_read_fn = pil_read_image
    elif image_reader == 'accimage':
        image_read_fn = accimage_read_image
    else:
        raise ValueError("Unknown image reading function")

    if video_format and video_format.lower() == 'video':
        return VideoFileReader()
    return ImageDirReader(image_read_fn)


def read_flow(path, frame_indices):
    video = []
    for i in frame_indices[:-1]:
        flow_x_path = os.path.join(path, 'flow_x_{:05d}.jpg'.format(i))
        flow_y_path = os.path.join(path, 'flow_y_{:05d}.jpg'.format(i))
        if os.path.exists(flow_x_path) and os.path.exists(flow_y_path):
            with open(flow_x_path, 'rb') as f:
                with Image.open(f) as img:
                    flow_x = img.convert('L')
            with open(flow_y_path, 'rb') as f:
                with Image.open(f) as img:
                    flow_y = img.convert('L')
            video.append(flow_x)
            video.append(flow_y)
        else:
            raise Exception(flow_x_path + " does not exist")
    return video
