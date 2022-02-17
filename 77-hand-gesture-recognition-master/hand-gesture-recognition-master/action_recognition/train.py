import torch
from torch.optim import lr_scheduler

from .utils import (calculate_accuracy, get_predictions, prepare_batch,
                    save_checkpoint)
from .validation import validate
import torchvision

def train_epoch(args, epoch, data_loader, model, criterion, optimizer, logger, decoder=None):
    print('train at epoch {}'.format(epoch))
    model.train()

    for i, (inputs_dict, targets) in logger.scope_enumerate(data_loader, epoch, total_time='time/train_epoch',
                                                            fetch_time='time/train_data', body_time='time/train_step'):

        grid = torchvision.utils.make_grid(inputs_dict['rgb_clip'][0].add(2.).div(8.).clamp(0., 1.))
        args.writer.add_image('rgb_clip_train', grid, 0)
        #if 'flow_clip' in inputs_dict:
        #    grid = torchvision.utils.make_grid(inputs_dict['flow_clip'][0])
        #    args.writer.add_image('flow_clip', grid, 0)

        args.writer.close()

        batch_size, inputs, labels = prepare_batch(args, inputs_dict, targets)
        outputs = model(*inputs)

        loss = criterion(outputs=outputs.cpu(), inputs=inputs, targets=labels.cpu())
        acc = calculate_accuracy(get_predictions(outputs, decoder), labels)

        if i % args.iter_size == 0:
            optimizer.zero_grad()

        loss.backward()

        if args.gradient_clipping:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.gradient_clipping)

        if args.iter_size > 1 and (i + 1) % args.iter_size == 0:
            for p in model.parameters():
                p.grad.data.mul_(1 / args.iter_size)

        optimizer.step()

        logger.log_value('train/loss', loss.item(), batch_size)
        logger.log_value('train/acc', acc, batch_size)
        if 'kd' in criterion.values:
            logger.log_value("train/kd_loss", criterion.values['kd'].item())

    logger.log_value("train/epoch_loss", logger.get_value("train/loss"))
    logger.log_value("train/epoch_acc", logger.get_value("train/acc"))

    return logger.get_value("train/acc"), logger.get_value("train/loss")


def train(args, model, train_loader, val_loader, criterion, optimizer, scheduler, logger, decoder=None):

    best_val_acc = 0.0
    for epoch in range(args.begin_epoch, args.n_epochs + 1):
        with logger.scope(epoch):
            for i, group in enumerate(optimizer.param_groups):
                group_name = group.get('group_name', i)
                logger.log_value("lr/{}".format(group_name), group['lr'])

        with logger.scope(epoch):
            train_acc, loss = train_epoch(args, epoch, train_loader, model, criterion, optimizer, logger, decoder)

        with logger.scope(epoch):
            val_acc = validate(args, epoch, val_loader, model, criterion, logger, decoder)
            logger.log_value("val/generalization_error", val_acc - train_acc)

        if args.checkpoint > 0:
            if epoch % args.checkpoint == 0:
                checkpoint_name = 'save_{}.pth'.format(epoch)
                save_checkpoint(checkpoint_name, model, optimizer, epoch, args)
        else:
            # save highest clip accuracy only
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_name = 'save_best.pth'
                save_checkpoint(checkpoint_name, model, optimizer, epoch, args)

        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_acc)
        else:
            scheduler.step()

        logger.reset_values('train')
        logger.reset_values('val')
