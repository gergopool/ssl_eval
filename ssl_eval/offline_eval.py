import torch
from torch import nn
from typing import List, Union
from functools import cached_property

from .generator import EmbGenerator
from .lin_eval import LinearEvaluator
from .knn import KNNEvaluator

__all__ = ['OfflineEvaluator']


class OfflineEvaluator:
    """OfflineEvaluator
    Evaluator that handles evaluation after a training epochs.

    Parameters
    ----------
    model : nn.Module
        The encoder model.
    dataset : str
        Name of the dataset. Choose from 'imagenet', 'cifar10' and 'cifar100'.
    root : str
        The path to the dataset's root.
    n_views : int, optional
        Number of desired views on train images. By default 1
    batch_size : int, optional
        Batch size for embedding generation, by default 256
    verbose : bool, optional
         Whether to print results to standard output or not. Only rank0 process
        prints. By default True
    """

    def __init__(self,
                 model: nn.Module,
                 dataset: str,
                 root: str,
                 n_views: int = 1,
                 batch_size: int = 256,
                 verbose: bool = True):
        self.model = model
        self.dataset = dataset
        self.emb_generator = EmbGenerator(model, dataset, root, n_views, batch_size, verbose)
        self.embs = None
        self.verbose = verbose

    @property
    def device(self):
        """Device of encoder model"""
        return next(self.model.parameters()).device

    @cached_property
    def cnn_dim(self):
        """Output dimension of encoder"""
        shape = (3, 244, 244) if self.dataset == 'imagenet' else (3, 32, 32)
        fake_input = torch.zeros(1, *shape).to(self.device)
        x = self.model(fake_input)
        return len(x[0])

    @cached_property
    def n_classes(self):
        """Number of classes of this dataset"""
        if self.dataset == "imagenet":
            return 1000
        elif self.dataset == "cifar10":
            return 10
        elif self.dataset == "cifa100":
            return 100
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def generate_embeddings(self):
        """Generatre train and validation embeddings"""
        self.embs = self.emb_generator()
        return self.embs

    def linear_eval(self,
                    embs: Union[None, List[torch.Tensor]] = None,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 0.1,
                    warm_start: bool = True) -> torch.Tensor:
        """linear_eval
        Runs a linear evaluation on pre-generated dataset.

        Parameters
        ----------
        embs : Union[None, List[torch.Tensor]], optional
            The embeddings in form (train_z, train_y, val_z, val_y).
            If not provided, it tries to reach the last generated ones. By default None
        epochs : int, optional
           Number of epochs, by default 100
        batch_size : int, optional
            Batch size, by default 256
        lr : float, optional
            Learning rate for LARC, by default 0.1
        warm_start : bool, optional
            If True and there has been a previous training, it loads the weights
        from the last training. By default True

        Returns
        -------
        torch.Tensor
            Top1 accuracy.

        Raises
        ------
        ValueError
            If no embeddings defined in either way.
        """

        if embs is None and self.embs is None:
            raise ValueError(f"No embeddings defined.")
        elif embs is None:
            embs = self.embs

        evaluator = LinearEvaluator(self.cnn_dim,
                                    self.n_classes,
                                    device=self.device,
                                    verbose=self.verbose,
                                    warm_start=warm_start)
        accuracy = evaluator(*embs, epochs, batch_size, lr)
        return accuracy

    def knn(self,
            embs: Union[None, List[torch.Tensor]] = None,
            k: Union[int, List[int]] = 1) -> torch.Tensor:
        """knn
        Runs knn evaluation on embeddings.

        Parameters
        ----------
        embs : Union[None, List[torch.Tensor]], optional
            The embeddings in form (train_z, train_y, val_z, val_y).
            If not provided, it tries to reach the last generated ones. By default None
        k : Union[int, List[int]], optional
            Desired K values to use for the algorithm. By default 1

        Returns
        -------
        torch.Tensor
            Top1 accuracy for each k, respectively.

        Raises
        ------
        ValueError
            If no embeddings defined in either way.
        """
        if embs is None and self.embs is None:
            raise ValueError(f"No embeddings defined.")
        elif embs is None:
            embs = self.embs
            
        evaluator = KNNEvaluator(self.device, self.verbose)
        accuracies = evaluator(*embs, k)
        return accuracies