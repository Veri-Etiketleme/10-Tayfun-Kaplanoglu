import torch
import copy
from torch import nn

from action_recognition.models.motion import MotionNetwork

from ..utils import get_fine_tuning_parameters


class TwoStreamNetwork(nn.Module):
    """ Two video transformer networks (encoder + decoder) with logits fusion (averaging)
        One network takes RGB clips as its input, one takes RGBDiff or OF clips as its input
    """
    def __init__(self, model, motion_mode="rgbdiff", path="", motion_path=""):
        super().__init__()

        self.model = model
        self.motion_model = MotionNetwork(copy.deepcopy(model), motion_mode)

        if path and motion_path:
            self.load_separate_trained(path, motion_path)

    def load_separate_trained(self, path, motion_path):
        print("Loading model from: {}".format(path))
        checkpoint = torch.load(path.as_posix())
        self.model.load_checkpoint(checkpoint['state_dict'])

        print("Loading motion model from: {}".format(motion_path))
        motion_checkpoint = torch.load(motion_path.as_posix())
        self.motion_model.load_checkpoint(motion_checkpoint['state_dict'])

    def forward(self, clip=None, flow_clip=None):
        """Extract the image feature vectors."""
        if flow_clip is not None:
            motion_input = flow_clip
        else:
            motion_input = clip

        # remove first sequence element to match the motion clip length
        # B x T x C x H x W -> B x T-1 x C x H x W
        logits = self.model(clip[:, 1:clip.size(1), ...])
        logits_motion = self.motion_model(motion_input)

        return 0.5 * logits + 0.5 * logits_motion

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)
