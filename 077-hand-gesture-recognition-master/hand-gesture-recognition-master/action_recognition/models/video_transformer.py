import torch
from torch import nn as nn
from torch.nn import functional as F

from .backbone import make_encoder
from .modules import Identity, squash_dims, unsquash_dim
from .modules.self_attention import DecoderBlock, PositionEncoding
from ..utils import get_fine_tuning_parameters, load_state


class VideoTransformer(nn.Module):
    def __init__(self, embed_size, sequence_size, encoder='resnet34', n_classes=400, input_size=(224, 224), pretrained=True,
                 input_channels=3, num_layers=4, layer_norm=True, recurrent=False):
        super().__init__()

        # backbone
        encoder = make_encoder(encoder, input_size=input_size, pretrained=pretrained, input_channels=input_channels)
        self.resnet = encoder.features  # name is kept for compatibility with older checkpoints
        self.last_feature_size = encoder.features_shape[1]
        self.embed_size = embed_size

        if encoder.features_shape[0] != embed_size:
            self.reduce_conv = nn.Conv2d(encoder.features_shape[0], embed_size, 1)
        else:
            self.reduce_conv = Identity()

        self.sequence_size = sequence_size

        if recurrent:
            self.self_attention_decoder = RecurrentSelfAttentionDecoder(embed_size, 8, num_layers,
                                                           sequence_size, layer_norm=layer_norm)
        else:
            self.self_attention_decoder = SelfAttentionDecoder(embed_size, embed_size, [8] * num_layers,
                                                           sequence_size, layer_norm=layer_norm)
        self.fc = nn.Linear(embed_size, n_classes)
        self.dropout = nn.Dropout2d(0.8)

        self.init_weights()
        self.input_channels = input_channels
        self.input_size = input_size

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, rgb_clip):
        """Extract the image feature vectors."""
        # (B x T x C x H x W) -> (B*T x C x H x W)
        rgb_clip = squash_dims(rgb_clip, (0, 1))

        features = self.resnet(rgb_clip)
        features = self.reduce_conv(features)

        features = F.avg_pool2d(features, features.shape[2:])  # (B*T) x C
        features = unsquash_dim(features, 0, (-1, self.sequence_size))
        ys = self.self_attention_decoder(features[..., 0, 0])
        # ys = self.dropout(ys)
        ys = self.fc(ys)

        return ys

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)

    def load_checkpoint(self, state_dict):
        load_state(self, state_dict, 'fc')


# we made a copy to be backward compatible with order models. However this class can also be used with a single input stream
class VideoTransformerMultiStreamEncoder(nn.Module):

    def __init__(self, embed_size, sequence_size, encoder='resnet34', n_classes=400, input_size=(224, 224), pretrained=True,
                 pretrain_paths=[], input_channels=3, num_layers=4, layer_norm=True, recurrent=False, num_streams=1,
                 fusion_mode='sum'):
        super().__init__()

        self.num_streams = num_streams
        self.fusion_mode = fusion_mode
        self.encoders = nn.ModuleList()
        self.reduce_convs = nn.ModuleList()

        if fusion_mode == 'sum':
            # N x conv1x1 -> N x avg_pool -> sum
            reduction_size = embed_size
        elif fusion_mode == 'cat1':
            # N x conv1x1 -> N x avg_pool -> cat
            assert embed_size % num_streams == 0
            reduction_size = embed_size // num_streams
        elif fusion_mode == 'cat2':
            # N x Identity -> N x avg_pool -> cat -> linear (same as conv1x1 on 1x1 HxW resolution)
            reduction_size = -1
        else:
            raise RuntimeError('Fusion mode {} is unknown'.format(fusion_mode))

        reduce_fc_inputs = 0
        for i in range(num_streams):
            # backbone
            enc = make_encoder(encoder, input_size=input_size, pretrained=pretrained, input_channels=input_channels)
            self.encoders.append(enc.features)
            reduce_fc_inputs += enc.features_shape[0]

            if enc.features_shape[0] != reduction_size and reduction_size >= 0:
                self.reduce_convs.append(nn.Conv2d(enc.features_shape[0], reduction_size, 1))
            else:
                self.reduce_convs.append(Identity())

        self.reduce_fc = Identity()
        if fusion_mode == 'cat2':
            self.reduce_fc = nn.Linear(reduce_fc_inputs, embed_size)

        self.sequence_size = sequence_size

        if recurrent:
            self.self_attention_decoder = RecurrentSelfAttentionDecoder(embed_size, 8, num_layers,
                                                                        sequence_size, layer_norm=layer_norm)
        else:
            self.self_attention_decoder = SelfAttentionDecoder(embed_size, embed_size, [8] * num_layers,
                                                               sequence_size, layer_norm=layer_norm)
        self.fc = nn.Linear(embed_size, n_classes)
        self.dropout = nn.Dropout2d(0.8)

        self.init_weights()
        self.input_channels = input_channels
        self.input_size = input_size

        # TODO: handle pretrain paths

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)

    def forward(self, *clips):
        """Extract the image feature vectors."""
        embeddings = []
        assert len(clips) == self.num_streams

        for i, clip in enumerate(clips):
            # (B x T x C x H x W) -> (B*T x C x H x W)
            clip = squash_dims(clip, (0, 1))

            features = self.encoders[i](clip)
            features = self.reduce_convs[i](features)

            # (B*T) x C x H x W -> (B*T) x C x 1 x 1
            features = F.avg_pool2d(features, features.shape[2:])
            # (B*T) x C x 1 x 1 -> B x T x C
            embeddings.append(unsquash_dim(features, 0, (-1, self.sequence_size))[..., 0, 0])

        if self.fusion_mode == 'sum':
            # S * [B x T x C] -> S x B x T x C
            embeddings = torch.stack(embeddings)
            # S x B x T x C -> B x T x C
            embeddings = embeddings.sum(0)
        else:
            # S * [B x T x C] -> B x T x (S*C)
            embeddings = torch.cat(embeddings, 2)
            embeddings = self.reduce_fc(embeddings)

        ys = self.self_attention_decoder(embeddings)
        # ys = self.dropout(ys)
        ys = self.fc(ys)

        return ys

    def trainable_parameters(self):
        param_groups = [
            ('trainable', {'re': r''}),
        ]

        return get_fine_tuning_parameters(self, param_groups)

    def load_checkpoint(self, state_dict):
        load_state(self, state_dict, 'fc')


class VideoTransformerEncoder(VideoTransformer):
    def forward(self, rgb_frame):
        features = self.resnet(rgb_frame)
        features = self.reduce_conv(features)
        features = F.avg_pool2d(features, 7)
        return features

    def export_onnx(self, export_path):
        first_param = next(self.parameters())
        input_tensor = first_param.new_zeros(1, self.input_channels, self.input_size[0], self.input_size[1])
        with torch.no_grad():
            torch.onnx.export(self, (input_tensor,), export_path, verbose=True)


class VideoTransformerDecoder(VideoTransformer):
    def forward(self, features):
        ys = self.self_attention_decoder(features)
        ys = self.fc(ys)
        return ys.mean(1)

    def export_onnx(self, export_path):
        first_param = next(self.parameters())
        input_tensor = first_param.new_zeros(1, self.sequence_size, self.embed_size)
        with torch.no_grad():
            torch.onnx.export(self, (input_tensor,), export_path, verbose=True)


class SelfAttentionDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_heads, sequence_size, inner_hidden_factor=2, layer_norm=True):
        super().__init__()

        input_sizes = [hidden_size] * len(n_heads)
        input_sizes[0] = input_size
        hidden_sizes = [hidden_size] * len(n_heads)

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.layers = nn.ModuleList([
            DecoderBlock(inp_size, hid_size, hid_size * inner_hidden_factor, n_head, hid_size // n_head,
                         hid_size // n_head, layer_norm=layer_norm)
            for i, (inp_size, hid_size, n_head) in enumerate(zip(input_sizes, hidden_sizes, n_heads))
        ])

    def forward(self, x):
        outputs, attentions = [], []
        b, t, c = x.size()
        x = self.position_encoding(x)

        for layer in self.layers:
            x, attn = layer(x)

            outputs.append(x)
        return x


class RecurrentSelfAttentionDecoder(nn.Module):
    """Uses a single decoder block that is applied n_steps times on the embeddings to refine them.
    """
    def __init__(self, hidden_size, n_heads, n_steps, sequence_size, inner_hidden_factor=2, layer_norm=True):
        super().__init__()

        self.position_encoding = PositionEncoding(sequence_size, hidden_size)

        self.n_steps = n_steps
        self.decoder_block = DecoderBlock(hidden_size, hidden_size, hidden_size * inner_hidden_factor,
                                          n_heads, hidden_size // n_heads, hidden_size // n_heads, layer_norm=layer_norm)

    def forward(self, x):
        x = self.position_encoding(x)

        for i in range(self.n_steps):
            x, attn = self.decoder_block(x)

        return x
