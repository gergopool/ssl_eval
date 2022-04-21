import torch
from functools import cached_property
from typing import Tuple

from .eval import Evaluator
from .emb_memory_bank import EmbMemoryBank


class StoreEvaluator(Evaluator):
    """OnlineEvaluator
    Similar to OfflineEvaluator, but this one does not generate training
    samples. Instead, while you train your neural network it saves
    embeddings

    Parameters
    ----------
    storage_size : int, optional
        The size of the FIFO queue in steps of epochs.
        This means this number is scaled up by the size of the trianing set,
        which is predetermined by the name of your chosen dataset. By default 1.
    """

    def __init__(self, *args, storage_size: int = 1., **kwargs):
        super(StoreEvaluator, self).__init__(*args, *kwargs)
        self.memory_bank = EmbMemoryBank(self.model,
                                         self.cnn_dim,
                                         storage_size * self.n_train_samples)

    @cached_property
    def n_train_samples(self):
        """Number of training samples of this dataset"""
        options = {
            "imagenet": 1281167, "tiny_imagenet": 100000, "cifar10": 50000, "cifar100": 50000
        }
        if self.dataset in options:
            return options[self.dataset]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def update(self, x: torch.Tensor, y: torch.Tensor):
        self.memory_bank.update(x, y)

    def generate_train_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.memory_bank.x, self.memory_bank.y
        self.embs[0] = x
        self.embs[1] = y
        return x, y