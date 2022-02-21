import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .utils import load_weights_from_file, adapt_network_input_channels
from ..modules import Identity


__all__ = [
    'ShuffleNetV2', 'shufflenetv2x05', 'shufflenetv2x10',
    'shufflenetv2x15', 'shufflenetv2x20'
]


imagenet_model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


cifar_model_files = {
    'shufflenetv2_x0.5': './data/checkpoints/shufflenetv2x05_cifar10.pth',
    'shufflenetv2_x1.0': './data/checkpoints/shufflenetv2x10_cifar10.pth',
    'shufflenetv2_x1.5': './data/checkpoints/shufflenetv2x15_cifar10.pth',
    'shufflenetv2_x2.0': None,
}

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000, num_channels=3,
                 input_size=(224, 224)):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        self.feature_size = input_size
        if min(input_size) // 32 < 3:
            self.is_low_resolution_model = True
            first_layer_stride = 1
            self.maxpool = Identity()
        else:
            self.is_low_resolution_model = False
            first_layer_stride = 2
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.feature_size = tuple(in_size // 4 for in_size in input_size)

        input_channels = num_channels
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, first_layer_stride, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            self.feature_size = tuple(in_size // 2 for in_size in self.feature_size)
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.feature_channels = output_channels
        self.fc = nn.Linear(output_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = x.mean([2, 3])  # globalpool
        x = self.fc(x)
        return x


def load_pretrained_model(model, arch, average_pretrained_input_weights=True, **kwargs):

    if model.is_low_resolution_model:
        pretrained_weights = load_weights_from_file(cifar_model_files[arch])
    else:
        pretrained_weights = model_zoo.load_url(imagenet_model_urls[arch])

    adapt_network_input_channels(pretrained_weights, 'conv1.0.weight',
                                 model.conv1[0].in_channels,
                                 average_pretrained_input_weights)
    model.load_state_dict(pretrained_weights)

    return model


def _shufflenetv2(arch, pretrained, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, arch, **kwargs)

    return model


def shufflenetv2x05(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, 
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenetv2x10(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenetv2x15(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenetv2x20(pretrained=False, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
