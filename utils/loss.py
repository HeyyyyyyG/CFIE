import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

### Acknowledgement: https://github.com/DingKe/pytorch_workplace/blob/master/focalloss/loss.py

def one_hot(index, classes):
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = index.new(*size).fill_(0)
    index = index.view(*view)
    ones = 1.

    if isinstance(index, Variable):
        ones = Variable(index.new(index.size()).fill_(1))
        mask = Variable(mask, volatile=index.volatile)

    return mask.scatter_(1, index, ones)


class FocalLoss(nn.Module):

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        y = one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.sum()


def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        target = one_hot(target, predict.size(-1))
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class LiftedStructureLoss(nn.Module):
    def __init__(self, alpha=40, beta=2, margin=0.5, hard_mining=None,  **kwargs):
        super(LiftedStructureLoss, self).__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta
        self.hard_mining = hard_mining

    def forward(self, inputs, targets):
        n = inputs.size(0)
        sim_mat = torch.matmul(inputs, inputs.t())
        targets = targets
        loss = list()
        c = 0

        for i in range(n):
            pos_pair_ = torch.masked_select(sim_mat[i], targets==targets[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1)
            neg_pair_ = torch.masked_select(sim_mat[i], targets!=targets[i])

            pos_pair_ = torch.sort(pos_pair_)[0]
            neg_pair_ = torch.sort(neg_pair_)[0]

            if self.hard_mining is not None:
                
                neg_pair = torch.masked_select(neg_pair_, neg_pair_ + 0.1 > pos_pair_[0])
                pos_pair = torch.masked_select(pos_pair_, pos_pair_ - 0.1 <  neg_pair_[-1])
            
                if len(neg_pair) < 1 or len(pos_pair) < 1:
                    c += 1
                    continue 
            
                pos_loss = 2.0/self.beta * torch.log(torch.sum(torch.exp(-self.beta*pos_pair), dim=0))
                neg_loss = 2.0/self.alpha * torch.log(torch.sum(torch.exp(self.alpha*neg_pair)))

            else:  
                pos_pair = pos_pair_
                neg_pair = neg_pair_ 

                pos_loss = 2.0/self.beta * torch.log(torch.sum(torch.exp(-self.beta*pos_pair)))
                neg_loss = 2.0/self.alpha * torch.log(torch.sum(torch.exp(self.alpha*neg_pair)))

            if len(neg_pair) == 0:
                c += 1
                continue

            loss.append(pos_loss + neg_loss)
        loss = torch.stack(loss)
        loss = loss.sum()/n
        # loss = sum(loss)/n
        prec = float(c)/n
        mean_neg_sim = torch.mean(neg_pair_).item()
        mean_pos_sim = torch.mean(pos_pair_).item()
        # return loss, prec, mean_pos_sim, mean_neg_sim
        return loss