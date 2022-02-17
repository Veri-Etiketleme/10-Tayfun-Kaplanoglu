import torch
from torch import nn

from ..utils import get_fine_tuning_parameters


class RGBDiff(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, image):
        """
        Args:
            image (torch.Tensor):  (N x T x C x H x W)

        """
        diffs = []
        for i in range(1, image.size(self.dim)):
            prev = image.index_select(self.dim, image.new_tensor(i - 1, dtype=torch.long))
            current = image.index_select(self.dim, image.new_tensor(i, dtype=torch.long))
            diffs.append(current - prev)

        return torch.cat(diffs, dim=self.dim)


class RGBRGBDiff(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_diff = RGBDiff(1)

    def forward(self, image):
        """
        Args:
            image (torch.Tensor):  (N x T x C x H x W)

        """
        diff = self.rgb_diff(image)
        return torch.cat([image[:, 1:, ...], diff], dim=2)


class MotionNetwork(nn.Module):
    def __init__(self, model, mode="rgbdiff"):
        """
        model:  pre constructed model to use
        mode:   motion modality to use
        """
        super().__init__()

        if mode == "flow":
            self.transform = nn.Identity()
        elif mode == "rgbdiff":
            self.transform = RGBDiff()
        elif mode == "rgbrgbdiff":
            self.transform = RGBRGBDiff()
        else:
            raise Exception("Unsupported mode " + mode)

        self.model = model

    def forward(self, clip):
        clip = self.transform(clip)
        return self.model(clip)

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)
