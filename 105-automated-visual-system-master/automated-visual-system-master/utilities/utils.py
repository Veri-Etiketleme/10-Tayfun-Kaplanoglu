import codecs
import datetime
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
# Third-party modules
import cv2
# Get the logger
logger = logging.getLogger('{}.{}'.format(
    os.path.basename(os.path.dirname(__file__)), __name__))


class WriteImageError(Exception):
    """Raised when an image couldn't be saved to disk"""


# ref.: https://stackoverflow.com/a/12412153
#       https://stackoverflow.com/a/667754
def get_full_command_line():
    return "python {}".format(subprocess.list2cmdline(sys.argv))


def load_json(path, encoding='utf8'):
    try:
        with codecs.open(path, 'r', encoding) as f:
            data = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(e)
    else:
        return data


# Setup logging from JSON configuration file
def setup_logging(logging_filepath, main_dirpath=None):
    try:
        # Read YAML configuration file
        config_dict = load_json(logging_filepath)
        if main_dirpath is not None:
            filename = config_dict['handlers']['file']['filename']
            new_filename = os.path.join(main_dirpath, filename)
            config_dict['handlers']['file']['filename'] = new_filename
        # Update the logging config dict with new values from `config_dict`
        logging.config.dictConfig(config_dict)
    except OSError as e:
        raise OSError(e)
    except KeyError as e:
        raise KeyError(e)
    except ValueError as e:
        raise ValueError(e)
    else:
        return config_dict


# This creates a timestamped filename/foldername so we don't overwrite our good
# work, ref.: https://stackoverflow.com/a/16713796
def timestamped(fname, fmt='%Y%m%d-%H%M%S-{fname}'):
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


# Return "folder_path/basename" if no file exists at this path. Otherwise,
# sequentially insert "_[0-9]+" before the extension of `basename` and return the
# first path for which no file is present.
# ref.: https://github.com/na--/ebook-tools/blob/0586661ee6f483df2c084d329230c6e75b645c0b/lib.sh#L295
def unique_filename(folder_path, basename):
    stem = Path(basename).stem
    ext = Path(basename).suffix
    new_path = os.path.join(folder_path, basename)
    counter = 0
    while os.path.isfile(new_path):
        counter += 1
        # `new_path` already exists, trying with `counter`
        new_stem = '{}_{}'.format(stem, counter)
        new_path = os.path.join(folder_path, new_stem) + ext
    return new_path


# Return `folder_path` if no folder exists at this path. Otherwise, sequentially
# insert "_[0-9]+" before the end of `folder_path` and return the first path for
# which no folder is present.
def unique_foldername(folder_path):
    counter = 0
    while os.path.isdir(folder_path):
        counter += 1
        # `folder_path` already exists, trying with counter `counter`
        folder_path = '{}_{}'.format(folder_path, counter)
    return folder_path


def write_image(path, image, overwrite_image=True):
    if os.path.isfile(path) and not overwrite_image:
        raise WriteImageError("File '{}' already exists and `overwrite` is "
                              "False".format(path))
    else:
        cv2.imwrite(path, image)
