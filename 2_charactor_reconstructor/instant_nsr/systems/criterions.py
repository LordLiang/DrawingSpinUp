import torch


def binary_cross_entropy(input, target, reduction='mean'):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    loss =  -(target * torch.log(input) + (1 - target) * torch.log(1 - input))

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


def ranking_loss(error, penalize_ratio=0.7, extra_weights=None , type='mean'):
    error, indices = torch.sort(error)
    # only sum relatively small errors
    s_error = torch.index_select(error, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
    if extra_weights is not None:
        weights = torch.index_select(extra_weights, 0, index=indices[:int(penalize_ratio * indices.shape[0])])
        s_error = s_error * weights

    if type == 'mean':
        return torch.mean(s_error)
    elif type == 'sum':
        return torch.sum(s_error)