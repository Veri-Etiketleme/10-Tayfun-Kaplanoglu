import torch


def load_weights_from_file(weights_file):
    model_dict = torch.load(weights_file)

    if 'net' in model_dict:
        model_dict = model_dict['net']

    # remove 'module.' prefix is present in keynames
    model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}

    return model_dict


def adapt_network_input_channels(pretrained_params, input_layer_key, num_channels, average_input_weights=False):
    """
    If average_input_weights == True: num_channels can be any positive number
    else: num_channels must be a multiple of the number of pretrained input channels
    """

    input_weights = pretrained_params[input_layer_key]

    if average_input_weights:
        input_weights = torch.mean(input_weights, dim=1)
        repeat_count = num_channels
    else:
        num_pretrained_channels = input_weights.size(1)
        if num_channels % num_pretrained_channels != 0:
            raise RuntimeError("If average_input_weights is False, expecting num_channels to be a multiple " +
                                "of {}, got {}".format(num_pretrained_channels, num_channels))
        repeat_count = num_channels // num_pretrained_channels

    pretrained_params[input_layer_key] = input_weights.unsqueeze(1).repeat(1, repeat_count, 1, 1)


