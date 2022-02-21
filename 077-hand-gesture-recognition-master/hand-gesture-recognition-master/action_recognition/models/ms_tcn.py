import torch
from torch import nn
from torch.nn import functional as F

from ..utils import get_fine_tuning_parameters
from .backbone import make_encoder
from .modules import (squash_dims, unsquash_dim)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MsTcn(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes, causal):
        super(MsTcn, self).__init__()
        self.conv_in = nn.Conv1d(dim, num_f_maps, 1)
        self.stages = nn.ModuleList([Tcn(num_layers, num_f_maps, causal[s]) for s in range(num_stages)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        x = self.conv_in(x)
        for stage in self.stages:
            x = stage(x)
        x = self.conv_out(x)
        return x


class Tcn(nn.Module):
    def __init__(self, num_layers, num_f_maps, causal):
        super(Tcn, self).__init__()
        self.layers = nn.ModuleList([DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, causal[i])
                                     for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, causal=False):
        super(DilatedResidualLayer, self).__init__()
        if causal:
            self.conv_dilated = CausalConv1d(in_channels, out_channels, 3, dilation=dilation)
        else:
            self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class MultiStageTemporalConvNet(nn.Module):
    """Multi Stage Temporal Convolution Network (https://arxiv.org/pdf/1903.01945.pdf)"""

    def __init__(self, embed_size, encoder='resnet34', n_classes=400, input_size=(224, 224), pretrained=True,
                 input_channels=3, num_stages=1, num_layers=5, num_f_maps=0, causal_config='none'):
        super().__init__()

        # encoder
        encoder = make_encoder(encoder, input_size=input_size, pretrained=pretrained, input_channels=input_channels)
        self.encoder = encoder.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # for backward compatibility
        if num_f_maps > 0:
            embed_size = num_f_maps

        # create causal flags for all layers of the network based on predefined config words
        if causal_config == 'none':
            # default: no causal residual conv blocks are used
            causal = [[False] * num_layers] * num_stages
        elif causal_config == 'all':
            # all residual conv blocks are causal
            causal = [[True] * num_layers] * num_stages
        elif causal_config.startswith('mix_'):
            # treat all stages the same. The number after generic gives the number of non-causal layers,
            # starting with thre first layers of every stage.
            nc_layers = int(causal_config[4:])
            causal = [[False] * nc_layers + [True] * (num_layers - nc_layers)] * num_stages
        else:
            raise RuntimeError("Unknown causal config")

        # decoder
        self.decoder = MsTcn(num_stages, num_layers, embed_size, encoder.features_shape[0], n_classes, causal)

    def forward(self, images):
        """Extract the image feature vectors and run them through the MS-TCN"""
        batch_size = images.size(0)

        # (B x T x C x H x W) -> (B*T x C x H x W)
        images = squash_dims(images, (0, 1))
        features = self.encoder(images)
        features = self.avgpool(features)
        # (B*T x C x 1 x 1) -> (B x T x C x 1 x 1)
        features = unsquash_dim(features, 0, (batch_size, -1))
        # (B x T x C x 1 x 1) -> (B x C x T)
        features = features.squeeze(-1).squeeze(-1).transpose(1, 2)

        ys = self.decoder(features)
        # (B x n_classes x T) -> (B x T x n_classes)
        ys = ys.transpose(1, 2)
        return ys

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)
