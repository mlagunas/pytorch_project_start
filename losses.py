import torch.nn as nn


def _apply_reduction(t, reduction):
    if reduction == 'mean':
        return t.mean()
    if reduction == 'batch-mean':
        return t.sum(dim=1).mean()
    if reduction == 'sum':
        return t.sum()
    if reduction == 'none':
        return t


_reductions = ['mean', 'sum', 'batch-mean', 'none']


class SquaredDifferences(nn.Module):
    def __init__(self, reduction='mean'):
        super(SquaredDifferences, self).__init__()

        ## store the reduction to apply to the loss
        assert reduction in _reductions
        self.reduction = reduction

    def forward(self, pred, real):
        diff = (pred - real) ** 2

        return _apply_reduction(diff, self.reduction)
