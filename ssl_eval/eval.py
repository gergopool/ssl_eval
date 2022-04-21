import torch
from torch import nn
from typing import List, Union, Tuple, Callable
from functools import cached_property

from .generator import EmbGenerator
from .lin_eval import LinearEvaluator
from .knn import KNNEvaluator
from .snn import SNNEvaluator

__all__ = ['Evaluator']


class Evaluator:
    """Evaluator
    Evaluator that handles evaluation after a training epochs.

    Parameters
    ----------
    model : nn.Module
        The encoder model.
    dataset : str
        Name of the dataset. Choose from 'imagenet', 'tiny_imagenet', 'cifar10' and 'cifar100'.
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
                 verbose: bool = True,
                 train_transform: Callable = None,
                 val_transform: Callable = None):
        self.model = model
        self.dataset = dataset
        self.emb_generator = EmbGenerator(model,dataset, root, n_views, batch_size,
                                          verbose, train_transform, val_transform)
        self.embs = None
        self.verbose = verbose

        self.embs = [None, None, None, None]

    @property
    def device(self):
        """Device of encoder model"""
        return next(self.model.parameters()).device

    @property
    def any_embs_none(self):
        for emb in self.embs:
            if emb is None:
                return True
        return False

    @cached_property
    def cnn_dim(self):
        was_training = self.model.training
        self.model.eval()
        shape = (3,32,32)
        fake_input = torch.zeros(1, *shape).to(self.device)
        x = self.model(fake_input)
        self.model.train(was_training)
        return len(x[0])

    @cached_property
    def n_classes(self):
        """Number of classes of this dataset"""
        options = {"imagenet": 1000, "cifar10": 10, "cifar100": 100, 'tiny_imagenet': 200}
        if self.dataset in options:
            return options[self.dataset]
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

    def generate_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generatre train and validation embeddings"""
        train_embs = self.generate_train_embeddings()
        val_embs = self.generate_val_embeddings()
        return *train_embs, *val_embs

    def generate_train_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z,y = self.emb_generator.get_train_embs()
        self.embs[0] = z
        self.embs[1] = y
        return z, y

    def generate_val_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        z,y = self.emb_generator.get_val_embs()
        self.embs[2] = z
        self.embs[3] = y
        return z, y

    def linear_eval(self,
                    embs: Union[None, List[torch.Tensor]] = None,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 0.1,
                    warm_start: bool = False) -> torch.Tensor:
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
        from the last training. By default False

        Returns
        -------
        torch.Tensor
            Top1 accuracy.

        Raises
        ------
        ValueError
            If no embeddings defined in either way.
        """

        if embs is None and self.any_embs_none:
            raise ValueError(f"Some embeddings are not defined.")
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
        if embs is None and self.any_embs_none:
            raise ValueError(f"Some embeddings are not defined.")
        elif embs is None:
            embs = self.embs

        evaluator = KNNEvaluator(self.device, self.verbose)
        accuracies = evaluator(*embs, k)
        return accuracies

    def snn(self,
            embs: Union[None, List[torch.Tensor]] = None,
            tau :float = 0.1,
            single_view: bool = True,
            balance_labels: bool = False) -> torch.Tensor:
        """knn
        Runs snn evaluation on embeddings.

        Parameters
        ----------
        embs : Union[None, List[torch.Tensor]], optional
            The embeddings in form (train_z, train_y, val_z, val_y).
            If not provided, it tries to reach the last generated ones. By default None
        tau : float
            Scale of embeddings before softmax.
        single_view : bool
            Use the first view only from the train set. Otherwise the different views
            act as different data points
        balance_labels : bool
            If the train data is not balanced, balance the weights of votes.

        Returns
        -------
        torch.Tensor
            Top1 accuracy

        Raises
        ------
        ValueError
            If no embeddings defined in either way.
        """
        if embs is None and self.any_embs_none:
            raise ValueError(f"Some embeddings are not defined.")
        elif embs is None:
            embs = self.embs

        evaluator = SNNEvaluator(self.device, self.verbose, tau=tau)
        accuracy = evaluator(*embs, single_view=single_view, balance_labels=balance_labels)
        return accuracy