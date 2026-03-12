from enum import Enum

class Task(Enum):
    MAE = 'mae'
    RECON = 'recon'
    MEAN_RECON = 'mean_recon'


def compute_loss(task, outputs, targets, mask=None):
    if mask is not None:
        outputs = outputs * mask
        targets = targets * mask
    if task == Task.MAE:
        return ((outputs - targets) ** 2).mean()
    elif task == Task.RECON:
        return ((outputs - targets) ** 2).sum()
    elif task == Task.MEAN_RECON:
        return ((outputs - targets) ** 2).mean(dim=1).mean()
    else:
        raise ValueError('Invalid task specified: {}'.format(task))
