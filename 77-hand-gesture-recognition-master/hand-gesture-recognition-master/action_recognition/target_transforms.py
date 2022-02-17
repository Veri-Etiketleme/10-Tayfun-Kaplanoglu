import torch

class Compose(object):
    """Compose multiple target transforms"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, target):
        dst = []
        for t in self.transforms:
            dst.append(t(target))
        return dst


class ClassLabel(object):
    """Returns video label and name. Used for training and validation."""

    def __init__(self, max_label_sequence_length=4):
        self.max_label_sequence_length = max_label_sequence_length

    def __call__(self, target):
        label = target['label']
        if isinstance(label, list):
            if len(label) > self.max_label_sequence_length:
                raise RuntimeError("Max label sequence length ({}) exceeded, got {}"
                                   .format(self.max_label_sequence_length, label))
            label += [-1] * (self.max_label_sequence_length - len(label))
            label = torch.LongTensor(label)

        return {
            'label': label,
            'video': target['video']
        }


class VideoID(object):
    """Returns video name. Used for video prediction."""

    def __call__(self, target):
        return {
            'label': target['label'],
            'video_id': target['video_id']
        }
