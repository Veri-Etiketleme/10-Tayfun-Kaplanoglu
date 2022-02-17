from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt

import torch

from .model import create_model
from .utils import load_state


class LogitKLDivLoss(nn.Module):
    """Kullback–Leibler divergence loss. Inputs predicted and ground truth logits.

    Args:
        T (float): Softmax temperature.
    """

    def __init__(self, T=1):
        super().__init__()
        self.T = T

    def forward(self, p_logits, q_logits, **kwargs):
        log_p = F.log_softmax(p_logits / self.T, dim=1)
        q = F.softmax(q_logits / self.T, dim=1)
        return F.kl_div(log_p, q, reduction='batchmean') * self.T ** 2


class DistillationLoss(nn.Module):
    """Knowledge distillation loss.

    Args:
        teacher_model (torch.nn.Module): Model that will be used for supervision.
        T (float): Softmax temperature.
    """

    def __init__(self, teacher_model, T=1):
        super().__init__()
        self.teacher_model = teacher_model
        self.kl_div = LogitKLDivLoss(T)

    def forward(self, outputs, inputs, **kwargs):
        """
        Args:
            outputs: Predicted student model logits
            inputs: Inputs that have been used to produce outputs.
        """
        with torch.no_grad():
            teacher_logits = self.teacher_model(*inputs)
        return self.kl_div(outputs, teacher_logits)


class SoftmaxLoss(nn.Module):
    """Classification loss"""

    def forward(self, outputs, targets, **kwargs):
        if targets.dim() != 1:
            raise RuntimeError("SoftmaxLoss requires 1D tensor dims for the targets, got {}".format(targets.shape))

        if outputs.dim() == 3:
            # in case we have a B x T x C tensor (T = time steps), average time steps to become a B x C tensor
            outputs = outputs.mean(1)
        return F.cross_entropy(outputs, targets)


class CTCLoss(nn.Module):
    """ Connectionist Temporal Classification loss """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.loss = nn.CTCLoss(reduction='mean', *args, **kwargs)

    def forward(self, outputs, targets, **kwargs):
        """
        outputs: BxTxC tensor
        targets: BxN, where N is the max sequence length and negative numbers are used to pad shorter
                 sequences to the length of N. Zero is the default blank symbol. The targets tensor cannot
                 contain blank symbols
        """
        # since the current CTCLoss implementation only properly supports empty sequences on CPU, switch to cpu
        # (https://github.com/pytorch/pytorch/issues/18215)
        #outputs = outputs.cpu()    # this does not work, needed to put it in train.py:27
        #targets = targets.cpu()

        if targets.dim() != 2:
            raise RuntimeError("CTCLoss targets requires BxN tensor dims for the targets, got {}".format(targets.shape))

        if outputs.dim() != 3:
            raise RuntimeError("CTCLoss function requires BxTxC tensor dims for the predictions, got {}"
                               .format(outputs.shape))

        # prepare inputs: BxTxC -> TxBxC and apply log(softmax(x)) on the C dimension
        log_probs = outputs.permute(1, 0, 2).log_softmax(2)
        # input length is vector of size B, filled with value T
        input_lengths = torch.full(size=(outputs.size(0),), fill_value=outputs.size(1), dtype=torch.long)
        # target length is vector of length B. Values are the number of non negative elements in dimension N
        target_lengths = (targets.size(1) - targets.lt(0).sum(1))

        return self.loss(log_probs, targets, input_lengths, target_lengths)


class CTFLoss(nn.Module):
    """ Connectionist Temporal Focal Loss https://arxiv.org/pdf/1904.10619.pdf """

    def __init__(self, gamma=1, sample_weighted=False, blank=0):
        """
        :param gamma: focus parameter (see paper)
        :param sample_weighted: if true, sample weighting used used, else class weighting is used
        :param blank: index of blank symbol
        """
        super().__init__()
        self.gamma = gamma
        self.sample_weighted = sample_weighted
        self.blank = blank

    def ctc_alignment_targets(self, log_probs, targets, input_lengths, target_lengths):
        """
        Calculate the pseudo CTC targets based on the network log(probs) and the ground truth targets
        Code from: https://github.com/vadimkantorov/ctc.git
        """
        _log_probs = log_probs.detach().requires_grad_()
        loss = F.ctc_loss(_log_probs, targets, input_lengths, target_lengths, blank=self.blank, reduction='sum')
        probs = _log_probs.exp()

        # to simplify API we inline log_softmax gradient, i.e. next two lines are equivalent to:
        # grad_logits, = torch.autograd.grad(loss, logits, retain_graph = True).
        # Gradient formula explained at:
        # https://stackoverflow.com/questions/35304393/trying-to-understand-code-that-computes-the-gradient-wrt-to-the-input-for-logsof
        grad_log_probs, = torch.autograd.grad(loss, _log_probs)
        grad_logits = grad_log_probs - probs * grad_log_probs.sum(dim=-1, keepdim=True)

        temporal_mask = (torch.arange(len(_log_probs), device=input_lengths.device, dtype=input_lengths.dtype).unsqueeze(
            1) < input_lengths.unsqueeze(0)).unsqueeze(-1)

        return (probs * temporal_mask - grad_logits).detach()

    def plot(self, probs, ce_alignment_targets):
        # T x B x C
        probs = probs.detach().numpy()
        ce_alignment_targets = ce_alignment_targets.numpy()

        fig, axs = plt.subplots(2, figsize=(15, 5))
        fig.suptitle("probs and targets", fontsize=14)
        for i in range(probs.shape[2]):
            axs[0].plot(ce_alignment_targets[:, 0, i], label=str(i))
            axs[0].set_ylabel('probs')
            axs[0].set_title('pseudo GT')
            axs[1].plot(probs[:, 0, i], label=str(i))
            axs[1].set_ylabel('probs')
            axs[1].set_xlabel('timesteps')
            axs[1].set_title('softmax probabilities')

        handles, labels = axs[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        plt.show()

    def calc_ctfl(self, log_probs, ce_alignment_targets, eps=1e-8):
        # TxBxC tensors
        probs = log_probs.exp()
        #self.plot(probs, ce_alignment_targets)
        # clamp to a small positive eps value to avoid zeros in resulting weighting tensor, which can introduce nan in
        # the gradient calculation when gamma < 1
        class_weighting_term = torch.pow(torch.clamp(torch.abs(probs - ce_alignment_targets), min=eps), self.gamma)
        if self.sample_weighted:
            # sample weighted
            loss = - torch.sum(torch.sum(class_weighting_term, 2) * torch.sum(ce_alignment_targets * log_probs, 2))
        else:
            # class weighted
            loss = - torch.sum(class_weighting_term * ce_alignment_targets * log_probs)

        return loss

    def forward(self, outputs, targets, **kwargs):
        """
        outputs: BxTxC tensor
        targets: BxN, where N is the max sequence length and negative numbers are used to pad shorter
                 sequences to the length of N. Zero is the default blank symbol. The targets tensor cannot
                 contain blank symbols
        """

        if targets.dim() != 2:
            raise RuntimeError("CTCLoss targets requires BxN tensor dims for the targets, got {}".format(targets.shape))

        if outputs.dim() != 3:
            raise RuntimeError("CTCLoss function requires BxTxC tensor dims for the predictions, got {}"
                               .format(outputs.shape))

        # prepare inputs: BxTxC -> TxBxC
        outputs = outputs.permute(1, 0, 2)
        log_probs = outputs.log_softmax(2)
        # input length is vector of size B, filled with value T
        input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long)
        # target length is vector of length B. Values are the number of non negative elements in dimension N
        target_lengths = (targets.size(1) - targets.lt(0).sum(1))
        # calculate cross entropy pseudo targets: TxBxC tensor
        ce_alignment_targets = self.ctc_alignment_targets(log_probs, targets, input_lengths, target_lengths)

        # cross entropy loss calculation based on pseudo targets
        return self.calc_ctfl(log_probs, ce_alignment_targets)


class WeightedSumLoss(nn.Module):
    """Aggregate multiple loss functions in one weighted sum."""

    def __init__(self, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.values = {}

    def forward(self, outputs, **kwargs):
        total_loss = outputs.new(1).zero_()
        for loss in self.losses:
            loss_val = self.losses[loss](outputs=outputs, **kwargs)
            total_loss += self.weights[loss] * loss_val
            self.values[loss] = loss_val

        if self.normalize:
            total_loss /= sum(self.weights.values())

        return total_loss

    def add_loss(self, name, loss, weight=1.0):
        self.weights[name] = weight
        self.losses.add_module(name, loss)


def create_criterion(args):
    criterion = WeightedSumLoss()

    if args.ctc_loss:
        # TODO: create unified CTC loss function
        if args.ctc_loss_type == 'ctc':
            ctc = CTCLoss()
        elif args.ctc_loss_type == 'ctfl_cs':
            ctc = CTFLoss(args.ctc_gamma, sample_weighted=False)
        elif args.ctc_loss_type == 'ctfl_sp':
            ctc = CTFLoss(args.ctc_gamma, sample_weighted=True)
        else:
            raise NotImplementedError("No implementation for {}".format(args.ctc_loss_type))

        criterion.add_loss('ctc', ctc)
    else:
        softmax = SoftmaxLoss()
        criterion.add_loss('softmax', softmax)

    if args.teacher_model:
        teacher_model, _ = create_model(args, args.teacher_model)

        checkpoint = torch.load(str(args.teacher_checkpoint))
        load_state(teacher_model, checkpoint['state_dict'])
        teacher_model.eval()

        distillation_loss = DistillationLoss(teacher_model, T=8)
        criterion.add_loss(distillation_loss, 0.4)

    return criterion
