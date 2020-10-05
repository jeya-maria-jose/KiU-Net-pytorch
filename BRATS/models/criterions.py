import torch.nn.functional as F
import torch

# for metric code: https://github.com/MIC-DKFZ/BraTS2017/blob/master/utils_validation.py

cross_entropy = F.cross_entropy

def hard_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    Np, Nn = pos.numel(), neg.numel()

    pos = pos.sum()

    k = min(Np*alpha, Nn)
    if k > 0:
        neg, _ = torch.topk(neg, int(k))
        neg = neg.sum()
    else:
        neg = 0.0

    loss = (pos + neg)/ (Np + k)

    return loss


def hard_per_im_cross_entropy(output, target, alpha=3.0):
    n, c = output.shape[:2]
    output = output.view(n, c, -1)
    target = target.view(n, -1)

    mtx = F.cross_entropy(output, target, reduce=False)

    pos = target > 0
    num_pos = pos.long().sum(dim=1, keepdim=True)

    loss = mtx.clone().detach()
    loss[pos] = 0
    _, loss_idx = loss.sort(1, descending=True)
    _, idx_rank = loss_idx.sort(1)

    num_neg = torch.clamp(alpha*num_pos, max=pos.size(1)-1)
    neg = idx_rank < num_neg

    return mtx[neg + pos].mean()


def focal_loss(output, target, alpha=0.25, gamma=2.0):
    n = target.size(0)

    lsfm = F.cross_entropy(output, target, reduce=False)

    pos = (target > 0).float()
    Np  = pos.view(n, -1).sum(1).view(n, 1, 1, 1)

    Np  = torch.clamp(Np, 1.0)
    z   = pos * alpha / Np / n  + (1.0 - pos) * (1.0 - alpha) / Np / n
    z   = z.detach()

    focal = torch.pow(1.0 - torch.exp(-lsfm), gamma) * lsfm * z

    return focal.sum()


def mean_cross_entropy(output, target, alpha=3.0):
    mtx = F.cross_entropy(output, target, reduce=False)

    bg = (target == 0)

    neg = mtx[bg]
    pos = mtx[1-bg]

    pos = pos.mean() if pos.numel() > 0 else 0
    neg = neg.mean() if pos.neg() > 0 else 0

    loss = (neg * alpha + pos)/(alpha + 1.0)
    return loss




eps = 0.1
def dice(output, target):
    num = 2*(output*target).sum() + eps
    den = output.sum() + target.sum() + eps
    return 1.0 - num/den


def cross_entropy_dice(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 5):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice(o, t)

    return loss

# in original paper: class 3 is ignored
# https://github.com/MIC-DKFZ/BraTS2017/blob/master/dataset.py#L283
# dice score per image per positive class, then aveg
def dice_per_im(output, target):
    n = output.shape[0]
    output = output.view(n, -1)
    target = target.view(n, -1)
    num = 2*(output*target).sum(1) + eps
    den = output.sum(1) + target.sum(1) + eps
    return 1.0 - (num/den).mean()

def cross_entropy_dice_per_im(output, target, weight=1.0):
    loss = weight * F.cross_entropy(output, target)
    output = F.softmax(output, dim=1)
    for c in range(1, 5):
        o = output[:, c]
        t = (target==c).float()
        loss += 0.25*dice_per_im(o, t)

    return loss
