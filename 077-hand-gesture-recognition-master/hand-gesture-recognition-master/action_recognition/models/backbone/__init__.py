from collections import namedtuple

from torch import nn

from . import resnet
from . import mobilenetv2
from . import shufflenetv2
from . import mnasnet
from . import squeezenet
from . import rmnet

Encoder = namedtuple('Encoder', ('model', 'features', 'features_shape'))


def make_encoder(name, input_size=(224, 224), input_channels=3, pretrained=True):
    """Make encoder (backbone) with a given name and parameters"""

    features_size = tuple(in_size // 32 for in_size in input_size)
    num_features = 2048
    if name.startswith('resnet') or name.startswith('resnext'):
        model = getattr(resnet, name)(pretrained=pretrained, num_channels=input_channels, input_size=input_size)
        features = nn.Sequential(*list(model.children())[:-2])  # all layers except the AvgPool and FC layer
        features_size = model.feature_size
        num_features = model.feature_channels
    elif name.startswith('mnasnet'):
        model = getattr(mnasnet, name)(pretrained=pretrained)
        features = model.layers
        num_features = 1280
    elif name.startswith('mobilenetv2'):
        model = mobilenetv2.mobilenet_v2(pretrained=pretrained, num_channels=input_channels, input_size=input_size)
        features = model.features
        features_size = model.feature_size
        num_features = model.feature_channels
    elif name.startswith('shufflenetv2'):
        model = getattr(shufflenetv2, name)(pretrained=pretrained, num_channels=input_channels, input_size=input_size)
        features = nn.Sequential(*list(model.children())[:-1])
        num_features = 1024
    elif name.startswith('squeezenet'):
        model = getattr(squeezenet, name)(pretrained=pretrained, num_channels=input_channels, input_size=input_size)
        features = model.features
        features_size = model.feature_size
        num_features = model.feature_channels
    elif name.startswith('rmnet'):
        model = rmnet.RMNetClassifier(1000, pretrained=None)
        features = nn.Sequential(*list(model.children())[:-2])
        num_features = 512
    elif name.startswith('se_res'):
        model = load_from_pretrainedmodels(name)(pretrained='imagenet' if pretrained else None)
        features = nn.Sequential(*list(model.children())[:-2])
    else:
        raise KeyError("Unknown model name: {}".format(name))

    features_shape = (num_features, features_size[0], features_size[1])
    return Encoder(model, features, features_shape)


def load_from_pretrainedmodels(model_name):
    import pretrainedmodels
    return getattr(pretrainedmodels, model_name)
