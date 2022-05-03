import torch
from torch.utils.data import DataLoader
from .distributed import AllGather


class DistributedAverageMeter(object):
    """DistributedAverageMeter
    Accumulates a value while keeping track of the number of instances.
    In a distributed setup it also tracks these numbers accross all
    gpu instances.

    Parameters
    ----------
    device : torch.device
        The torch device. For distributed setup it must be a gpu.
    n : int, optional
        Number of tracked values. By default 1
    """

    def __init__(self, device: torch.device, n: int = 1):
        self.device = device
        self.n = n
        self.reset()

    @property
    def sum(self):
        return AllGather.apply(self._sum).sum(dim=0)

    @property
    def count(self):
        return AllGather.apply(self._count).sum(dim=0)

    @property
    def avg(self):
        return self.sum / self.count

    @property
    def local_avg(self):
        return self._sum[0] / self._count[0]

    def reset(self):
        self._sum = torch.zeros(1, self.n).to(self.device, non_blocking=True)
        self._count = torch.zeros(1, self.n).to(self.device, non_blocking=True)

    def update(self, val: torch.Tensor, n: int = 1):
        self._sum[0] += val
        self._count[0] += n


def iter_with_convert(data_loader: DataLoader, device: torch.device) -> torch.Tensor:
    next_x, next_y = None, None
    for (xs, y) in data_loader:
        out_x = next_x
        out_y = next_y
        if isinstance(xs, list):
            next_x = [x.to(device, non_blocking=True) for x in xs]
        elif isinstance(xs, torch.Tensor):
            next_x = xs.to(device, non_blocking=True)
        else:
            raise NotImplementedError
        next_y = y.to(device, non_blocking=True)
        if out_x is not None:
            yield out_x, out_y
    yield next_x, next_y