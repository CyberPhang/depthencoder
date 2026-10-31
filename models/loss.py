import torch.nn.functional as F

def gradient_loss(pred, target):
    loss = 0
    for _ in range(3):
        dp = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        dt = target[:, :, :, 1:] - target[:, :, :, :-1]
        loss = loss + F.l1_loss(dp, dt)
        dp = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dt = target[:, :, 1:, :] - target[:, :, :-1, :]
        loss = loss + F.l1_loss(dp, dt)
        pred = F.avg_pool2d(pred, 2)
        target = F.avg_pool2d(target, 2)
    return loss

