# Credits: facebookresearch
import os
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


# Code credits:
# https://github.com/facebookresearch/suncet/blob/main/src/utils.py
def init_distributed(port=40011, rank_and_world_size=(None, None)):

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    rank, world_size = rank_and_world_size
    os.environ['MASTER_ADDR'] = 'localhost'

    if (rank is None) or (world_size is None):
        try:
            world_size = int(os.environ['SLURM_NTASKS'])
            rank = int(os.environ['SLURM_PROCID'])
            os.environ['MASTER_ADDR'] = os.environ['HOSTNAME']
        except Exception:
            print('WARNING: Distributed training not available')
            world_size, rank = 1, 0
            return world_size, rank

    try:
        # Open a random port
        os.environ['MASTER_PORT'] = str(port)
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
    except Exception:
        world_size, rank = 1, 0
        print('WARNING: Distributed training not available')

    return world_size, rank


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
