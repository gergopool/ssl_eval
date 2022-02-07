import torch
from torch import distributed as dist

__all__ = ['get_world_size_n_rank', 'AllGether', 'AllReduce']


def get_world_size_n_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    else:
        return 1, 0


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads


class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (dist.is_available() and dist.is_initialized() and (dist.get_world_size() > 1)):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads
