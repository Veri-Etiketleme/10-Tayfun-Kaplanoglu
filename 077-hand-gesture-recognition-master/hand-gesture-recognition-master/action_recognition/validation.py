import torch
from .utils import AverageMeter, get_predictions, calculate_accuracy, prepare_batch
import torchvision


def validate(args, epoch, data_loader, model, criterion, logger, decoder=None):
    print('validation at epoch {}'.format(epoch))
    model.eval()

    video_acc_avg = AverageMeter()

    clip_logits = []
    previous_video_id = None
    previous_video_gt = None
    for i, (inputs, targets) in logger.scope_enumerate(data_loader, epoch, total_time='time/val_epoch',
                                                       fetch_time='time/val_data', body_time='time/val_step'):

        #grid = torchvision.utils.make_grid(inputs['rgb_clip'][0].add(2.).div(4.).clamp(0., 1.))
        #args.writer.add_image('rgb_clip_val', grid, 0)

        video_ids = targets['video']
        batch_size, inputs, labels = prepare_batch(args, inputs, targets)
        with torch.no_grad():
            outputs = model(*inputs)

        # calculate validation loss
        loss = criterion(outputs=outputs.cpu(), targets=labels.cpu(), inputs=inputs)

        # calculate clip and video accuracy
        video_acc = acc = calculate_accuracy(get_predictions(outputs, decoder), labels)

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

        logger.log_value("val/loss", loss.item(), batch_size)
        logger.log_value("val/acc", acc, batch_size)
        logger.log_value("val/video", video_acc_avg.avg)

    return logger.get_value("val/acc")
