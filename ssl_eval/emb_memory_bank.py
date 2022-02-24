import torch
from torch import nn

from .distributed import AllGather


class EmbMemoryBank:
    """EmbMemoryBank
    This class stores the embeddings that are generated during
    a training epoch. It works like a FIFO, but uses an index-shift
    in order to save unnecesarry computation by copying tensors.

    Parameters
    ----------
    model : nn.Module
        The encoder model. It is later used to extract cnn_dim and device.
    cnn_dim : int
        Dimensionality of cnn encoder output.
    max_size : int, optional
        Maximum number of embeddings to store.
        By default 2000
    """

    def __init__(self, model: nn.Module, cnn_dim: int, max_size: int = 2000):

        self.model = model
        self.cnn_dim = cnn_dim
        self.max_size = max_size

        # This will store the last N representations seen
        self.memory_bank = torch.zeros((max_size, self.cnn_dim)).half().cpu()

        # Next index we should update per class in memory bank
        self.index = 0

        # Number of embeddings already stored in memory bank
        self.memory_size = 0

        # Labels of the memory bank
        self.labels = torch.zeros(max_size).long().cpu()

    @property
    def device(self):
        """Device of encoder model"""
        return next(self.model.parameters()).device

    @property
    def x(self):
        return self.memory_bank[:self.memory_size].unsqueeze(dim=-1)

    @property
    def y(self):
        return self.labels[:self.memory_size]

    def reset(self, should_reset=True):
        if should_reset:
            self.memory_size = 0

    def update(self, x: torch.Tensor, y: torch.Tensor):
        """Save new embeddings in memory along with their classes.
        Parameters
        ----------
        x : torch.Tensor
            The embeddings of images.
        y : torch.Tensor
            The labels of the embeddings.
        """
        x = AllGather.apply(x.to(self.device)).detach().half().cpu()
        y = AllGather.apply(y.to(self.device)).detach().long().cpu()
        n = len(y)

        assert n < self.max_size, "Your queue size is too small for this batch of data."

        if n + self.index < self.max_size:
            # Add data
            self.memory_bank[self.index:self.index + n] = x
            self.labels[self.index:self.index + n] = y
            # Update status
            self.index += n
            self.memory_size = min(self.memory_size + n, self.max_size)
        else:
            # Overflow happened
            # Number of spaces left in the end of table
            k = self.max_size - self.index

            # Fill end of table
            self.memory_bank[self.index:] = x[:k]
            self.labels[self.index:] = y[:k]

            # Fill beginning of table
            self.memory_bank[:n - k] = x[k:]
            self.labels[:n - k] = y[k:]

            # Update status
            self.index = n - k
            self.memory_size = self.max_size
