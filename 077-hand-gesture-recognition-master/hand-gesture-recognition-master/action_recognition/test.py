import torch
import torch.nn.functional as F

from .utils import AverageMeter, prepare_batch
from .utils import get_predictions, calculate_accuracy, plot_confusion_matrix, load_json
import matplotlib.pyplot as plt
import numpy as np

def test(args, data_loader, model, logger, decoder=None):
    print('test')
    model.eval()

    video_acc_avg = AverageMeter()

    clip_logits = []
    all_predictions = []
    all_labels = []
    previous_video_id = None
    previous_video_gt = None
    for i, (inputs, targets) in logger.scope_enumerate(data_loader):
        video_ids = targets['video']
        batch_size, inputs, labels = prepare_batch(args, inputs, targets)

        outputs = model(*inputs)

        if decoder is None and args.softmax_in_test:
            outputs = F.softmax(outputs)

        clip_pred = get_predictions(outputs, decoder)

        # store predicted labels and ground thruth labels to later calculate a confusion matrix
        all_predictions.append(clip_pred)
        if labels.dim() == 2:
            # in ctc mode, targets can be a sequence, assume the sequence contains only repeated labels of the same class
            # therefor, take the first element only. If empty (-1), replace with zero
            labels = labels[:, 0].cpu()
            labels = labels.clamp(min=0)
        all_labels.append(labels)

        video_acc = clip_acc = calculate_accuracy(clip_pred, labels)

        if decoder is None:
            # calculate video outputs by averaging clip outputs from the same video
            # we cannot do this in CTC mode
            video_outputs = []
            video_labels = []
            for j in range(batch_size):
                if video_ids[j] != previous_video_id and previous_video_id is not None:
                    clip_logits = torch.stack(clip_logits)
                    video_logits = torch.mean(clip_logits, dim=0)
                    video_outputs.append(video_logits)
                    video_labels.append(previous_video_gt)
                    clip_logits = []

                # outputs = BxTxC (ctc) or BxC (non ctc), clip_logits = CLIPxTxC (ctc) or CLIPxC (non ctc)
                clip_logits.append(outputs[j].data.cpu())
                previous_video_id = video_ids[j]
                previous_video_gt = labels[j].cpu()

            if video_outputs:
                video_outputs = torch.stack(video_outputs)
                video_labels = torch.stack(video_labels)
                video_acc = calculate_accuracy(get_predictions(video_outputs), video_labels)

        video_acc_avg.update(video_acc)

        logger.log_value("test/acc", clip_acc, batch_size)
        logger.log_value("test/video", video_acc_avg.avg)

    # save confusion matrix as image
    data = load_json(args.annotation_path)
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    plot_confusion_matrix(all_predictions, all_labels, data["labels"])
    plt.savefig(str(args.result_path / 'confusion_matrix.pdf'))

    return logger.get_value("test/video"), logger.get_value("test/acc")
