# Credits: facebookresearch

import torch
from torch import distributed as dist
from typing import Tuple

__all__ = ['get_world_size_n_rank', 'AllGether', 'AllReduce']

# ===================================================================
# Public functions
# ===================================================================


def get_world_size_n_rank() -> Tuple[int, int]:
    """get_world_size_n_rank
    Determines the number of processes and rank of current process.

    Returns
    -------
    Tuple[int, int]
        World size and rank, respectively.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


class AllGather(torch.autograd.Function):
    """AllGather
    If distributed running, concatenates the requested tensors
    along 0th dimension. If not, simply returns with original tensor.
    """

    @staticmethod
    def forward(_, x):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(_, grads):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(torch.autograd.Function):
    """AllReduce
    If distributed running, returns with the mean of tensors along the first dimension.
    If not, simply returns with original tensor.
    """

    @staticmethod
    def forward(_, x):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(_, grads):
        return grads
