import json
import operator
import pickle
import re
import subprocess
import sys
import tarfile
from collections import OrderedDict
from hashlib import md5
from itertools import islice, tee
from pathlib import Path

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from action_recognition.options import get_argument_parser


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def load_value_file(file_path):
    with open(str(file_path), 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


def load_json(data_file_path):
    with open(str(data_file_path), 'r') as data_file:
        return json.load(data_file)


def get_predictions(outputs, ctc_decoder=None):

    if ctc_decoder is not None:
        if outputs.dim() != 3:
            raise RuntimeError("CTC beam search decoder requires BxTxC tensor dims")

        probs = outputs.softmax(2)
        labels, label_scores = ctc_decoder.decode(probs)
        # calculate single prediction for the whole time sequence by taking the label in the decoded sequence with the
        # highest score
        _, max_indices = label_scores.max(1)
        predictions = labels[torch.arange(labels.size(0)), max_indices]
        # empty predictions (-1) belong to the non action class, so rename -1 to the blank symbol
        predictions = predictions.clamp(min=0).cpu()

    else:
        if outputs.dim() == 3:
            # in case we have a B x T x C tensor (T = time steps), average time steps to become a B x C tensor
            outputs = outputs.mean(1)

        # calculate correct classifications
        _, pred = outputs.topk(1, 1, True)
        predictions = pred.t()[0].cpu()

    return predictions


def plot_confusion_matrix(predictions, targets, class_names, normalize=True, cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    predictions = predictions.numpy().astype(np.int)
    targets = targets.cpu().numpy().astype(np.int)
    class_names = np.array(class_names)

    # Compute confusion matrix
    cm = confusion_matrix(targets, predictions)
    # Only use the labels that appear in the data
    class_names = class_names[unique_labels(targets, predictions)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=4)
    fig.tight_layout()
    return ax


def calculate_accuracy(predictions, targets):
    batch_size = targets.size(0)

    if targets.dim() == 2:
        # in ctc mode, targets is a BxT tensor. Assume T contains only repeated labels of the same class.
        # Therefor, take the first element only. If T is empty (-1 at first position), replace with zero
        targets = targets[:, 0]
        targets = targets.clamp(min=0)

    correct = predictions.eq(targets.cpu())
    n_correct_elems = correct.float().sum().item()
    return n_correct_elems / batch_size


def mkdir_if_not_exists(path):
    Path(path).mkdir(exist_ok=True, parents=True)


class TeedStream(object):
    """Copy stdout to the file"""

    def __init__(self, fname, mode='w'):
        self.file = open(str(fname), mode)
        self.stdout = sys.stdout
        sys.stdout = self

    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


def get_nested_attr(obj, attr_name):
    """Same as getattr(obj, attr_name), but allows attr_name to be nested
    e.g. get_nested_attr(obj, "foo.bar") is equivalent to obj.foo.bar"""
    for name_part in attr_name.split('.'):
        if hasattr(obj, name_part):
            obj = getattr(obj, name_part)
        else:
            raise AttributeError("module has no " + name_part + " attribute")

    return obj


def load_state(module, state_dict, classifier_layer_name=None, remaps=None):
    """
    Robust checkpoint loading routine.

    Args:
        module: Loaded model
        state_dict: dict containing parameters and buffers
        classifier_layer_name (str): name of the classifier layer of the model
        remaps (dict): mapping from checkpoint names to module names
    """
    if isinstance(module, nn.DataParallel):
        module = module.module

    if remaps is None:
        remaps = {}

    # unwrap module in state dict
    unwrapped_state = OrderedDict()
    strip_line = 'module.'
    for k, v in state_dict.items():
        if k.startswith(strip_line):
            k = k[len(strip_line):]

        if k in remaps:
            k = remaps[k]

        unwrapped_state[k] = v

    if classifier_layer_name is None:
        return module.load_state_dict(unwrapped_state, strict=False)

    module_classes = get_nested_attr(module, classifier_layer_name).out_features
    checkpoint_classes = unwrapped_state['{}.weight'.format(classifier_layer_name)].size(0)

    if module_classes != checkpoint_classes:
        print("Number of classes in model and checkpoint vary ({} vs {}). Do not loading last FC weights".format(
            module_classes, checkpoint_classes))
        del unwrapped_state['{}.weight'.format(classifier_layer_name)]
        del unwrapped_state['{}.bias'.format(classifier_layer_name)]

    return module.load_state_dict(unwrapped_state, strict=False)


def save_checkpoint(checkpoint_name, model, optimizer, epoch_no, args):
    save_file_path = args.result_path / 'checkpoints' / checkpoint_name
    states = {
        'epoch': epoch_no + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_file_path.as_posix())


def create_code_snapshot(root, dst_path, extensions=(".py", ".json")):
    """Creates tarball with the source code"""
    with tarfile.open(str(dst_path), "w:gz") as tar:
        for path in Path(root).rglob("*"):
            if '.git' in path.parts:
                continue
            if path.suffix.lower() in extensions:
                tar.add(path.as_posix(), arcname=path.relative_to(root).as_posix(), recursive=True)


def print_git_revision():
    try:
        rev = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).decode().strip()
        print("Git branch: {}".format(branch))
        print("Git rev: {}".format(rev))
    except subprocess.CalledProcessError:
        print("No git repo found")


def get_fine_tuning_parameters(model, param_groups=None):
    """Returns parameter groups in optimizer format. Allows to select per-layer learning rates."""
    if param_groups is None:
        param_groups = [('trainable', {'re': r''})]

    for param_name, param in model.named_parameters():
        group = None
        for group_name, group in param_groups:
            # find first matched group for a given param
            if re.search(group.get('re', ''), param_name):
                break

        #print("{} -> {}".format(param_name, group_name))
        group.setdefault('params', []).append(param)

    # save group names for plotting lrs
    # and init params key for optimizer
    for param_name, param in param_groups:
        param['group_name'] = param_name
        param.setdefault('params', [])

    return [group for name, group in param_groups]


def drop_last(iterable, n=1):
    """Drops the last item of iterable"""
    t1, t2 = tee(iterable)
    return map(operator.itemgetter(0), zip(t1, islice(t2, n, None)))


def json_serialize(obj):
    """Serialization function for json.dump"""
    if isinstance(obj, Path):
        return str(obj)
    return obj.__dict__


def prepare_batch(args, inputs_dict, targets):
    """Converts dict returned from data loader to tuple of tensors and converts targets to tensors"""
    labels = targets['label']
    labels = labels.to(args.device)
    for key in inputs_dict:
        inputs_dict[key] = inputs_dict[key].to(args.device)
        if args.fp16:
            inputs_dict[key] = inputs_dict[key].to(torch.half)
        batch_size = inputs_dict[key].size(0)
    inputs = tuple(inputs_dict[k] for k in ('rgb_clip', 'flow_clip') if k in inputs_dict)
    return batch_size, inputs, labels


def md5_hash(obj):
    obj_serialize = json.dumps(obj, default=json_serialize)
    digest = md5(obj_serialize.encode())
    return digest.hexdigest()


def cached(file=None):
    """Cache returned value of a wrapped function to disk. Next call with the same arguments will result in loading
    the saved values."""

    def decorator(fn):
        nonlocal file
        if file is None:
            file = "{}.cache".format(fn.__name__)

        def wrapped(*args, **kwargs):
            data = {'args': None, 'kwargs': None, 'ret': None}
            args_hex = md5_hash((args, kwargs))[-8:]
            file_hex = Path("{!s}.{}".format(file, args_hex))
            if file_hex.exists():
                with file_hex.open('rb') as f:
                    data = pickle.load(f)

            if data['args'] != args or data['kwargs'] != kwargs:
                data['args'] = args
                data['kwargs'] = kwargs
                data['ret'] = fn(*args, **kwargs)

                with file_hex.open('wb') as f:
                    pickle.dump(data, f)

            return data['ret']

        return wrapped

    return decorator


def generate_args(*args, **kwargs):
    argparser = get_argument_parser()
    argv = list(args)
    for k, v in kwargs.items():
        key_format = "--{:s}".format(k.replace('_', '-'))
        no_key_format = "--no-{:s}".format(k.replace('_', '-'))
        if isinstance(v, bool):
            if v:
                argv.append(key_format)
            else:
                argv.append(no_key_format)
        else:
            argv.append(key_format)
            argv.append(str(v))

    return argparser.parse_known_args(argv)
