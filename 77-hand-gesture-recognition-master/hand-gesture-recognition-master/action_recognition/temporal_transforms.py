import random
import numpy as np
import collections


class LoopPadding(object):
    """Extend short clip to a given size"""

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalStride:
    """Skips frames with a given step. Increases effective temporal receptive field."""

    def __init__(self, stride=1):
        self.stride = stride

    def __call__(self, frame_indices):
        return frame_indices[::self.stride]


class TemporalBeginCrop(object):
    """Temporally crop the given frame indices at a beginning.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        out = frame_indices[:self.size]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.

    Args:
        size (int): Desired output size of the crop.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[int(begin_index):int(end_index)]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        return out


class TemporalRandomScale(object):
    """Scale the temporal domain by repeating/removing frame indices
        The final result is cropped/padded to keep the same number of frame_indices at the end
    """

    def __init__(self, scale = 0.0):
        """A scale < 1.0 shortens the video, a scale > 1.0 elongates the video"""

        self.scale_range = (1.0 - scale, 1.0 + scale)
        self.scale = None

    def __call__(self, frame_indices):

        if self.scale is None:
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
        else:
            scale = self.scale

        num_frame_indices = len(frame_indices)
        scaled_num_frame_indices = round(scale * num_frame_indices)

        # stretch/shrink
        indices = np.round(np.linspace(0, num_frame_indices - 1, num=scaled_num_frame_indices)).astype(np.int)
        scaled_frame_indices = [frame_indices[i] for i in indices]

        # crop/pad
        if scaled_num_frame_indices > num_frame_indices:
            # crop
            diff = (scaled_num_frame_indices - num_frame_indices)
            odd = diff % 2
            crop_len = diff // 2
            scaled_frame_indices = scaled_frame_indices[crop_len:-(crop_len + odd)]

        elif scaled_num_frame_indices < num_frame_indices:
            # pad
            diff = (num_frame_indices - scaled_num_frame_indices)
            odd = diff % 2
            pad_len = diff // 2
            scaled_frame_indices = [scaled_frame_indices[0]] * pad_len +\
                                    scaled_frame_indices +\
                                   [scaled_frame_indices[-1]] * (pad_len + odd)

        return scaled_frame_indices


class TemporalRandomShift(object):
    """Temporally shift the frame indices back and forth by a random offset.

    If shifting to the right, the last frame indices are rolled over to the start
    If shifting the the left, the first frame indices are rolled over to the end
    """

    def __init__(self, shift_range=0.1, mode='pad'):
        """Shift range is a % of the total clip length
        """
        self.shift_range = shift_range
        self.mode = mode

    def __call__(self, frame_indices):

        shift_range = round(self.shift_range * len(frame_indices))
        shift = random.randint(-shift_range, shift_range)
        if self.mode == 'rotate':
            queue = collections.deque(frame_indices)
            queue.rotate(shift)
            result = list(queue)
        elif self.mode == 'pad':
            if shift >= 0:
                result = shift * [frame_indices[0]] + frame_indices[:-(shift + 1)]
            else:
                result = frame_indices[(-shift):] + (-shift) * [frame_indices[-1]]

        return result
