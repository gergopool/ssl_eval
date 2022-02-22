import torch
from .distributed import AllGather


class DistributedAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, device):
        self.reset()

    @property
    def sum(self):
        return AllGather.apply(self._sum)

    @property
    def count(self):
        return AllGather.apply(self._count)

    @property
    def avg(self):
        return self.sum / self.count

    def reset(self):
        self._sum = torch.zeros(1).to(self.device)
        self._count = torch.zeros(1).to(self.device)

    def update(self, val, n=1):
        self._sum += val
        self._count += n