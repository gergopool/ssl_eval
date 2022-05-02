import torch
from torch import nn
from torch.utils.data import DataLoader
from functools import cached_property
from typing import Callable, Tuple

from . import pkbar
from .data import get_loaders_by_name
from .distributed import AllGather, get_world_size_n_rank


class EmbGenerator:
    """EmbGenerator
    Generates embeddings by running the encoder model on the images
    of the desired dataset.

    Parameters
    ----------
    model : nn.Module
        Torch encoder that translates a (BxCxNxN) image to (BxD) representations. 
    dataset : str
        Name of the dataset. Choose from 'imagenet', 'cifar10' and 'cifar100'.
    root : str
       The path to the dataset's root.
    n_views : int, optional
        Number of desired views of a single image in train generation. By default 1
    batch_size : int, optional
        Batch size (per process), by default 256
    verbose : bool, optional
        Whether to print results to standard output or not. Only rank0 process
        prints. By default True
    """

    def __init__(self,
                 model : nn.Module,
                 dataset : str,
                 root : str,
                 n_views : int = 1,
                 batch_size: int = 256,
                 verbose: bool = True,
                 train_transform: Callable = None,
                 val_transform: Callable = None):
        self.model = model
        self.dataset = dataset
        self.root = root
        self.n_views = n_views
        self.batch_size = batch_size
        self.world_size, self._rank = get_world_size_n_rank()
        self.verbose = verbose and self._rank == 0

        data_loaders = get_loaders_by_name(self.root,
                                           self.dataset,
                                           batch_size=self.batch_size,
                                           n_views=n_views,
                                           train_transform=train_transform,
                                           val_transform=val_transform)
        self.train_loader, self.val_loader = data_loaders

    @property
    def device(self):
        return next(self.model.parameters()).device

    @cached_property
    def cnn_dim(self):
        was_training = self.model.training
        self.model.eval()
        shape = (3,32,32)
        fake_input = torch.zeros(1, *shape).to(self.device)
        x = self.model(fake_input)
        self.model.train(was_training)
        return len(x[0])

    # =========================================================================
    # Public functions
    # =========================================================================

    def __call__(self):
        """ Generate both train and val embeddings """
        train_data = self.get_train_embs()
        val_data = self.get_val_embs()
        return *train_data, *val_data

    def get_val_embs(self):
        return self._generate(self.val_loader)

    def get_train_embs(self):
        return self._generate(self.train_loader)

    # =========================================================================
    # Private functions
    # =========================================================================

    def iter_with_convert(self, data_loader:DataLoader, device:torch.device) -> torch.Tensor:
        next_x, next_y = None, None
        for (xs, y) in data_loader:
            out_x = next_x
            out_y = next_y
            next_x = [x.to(device, non_blocking=True) for x in xs]
            next_y = y.to(device, non_blocking=True)
            if out_x:
                yield out_x, out_y
        yield next_x, next_y

    def _generate(self, data_loader:DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """_generate
        Runs over data data loader once and saves every generated embedding
        to the cpu. The embeddings are calcualted in half-precision.

        Parameters
        ----------
        data_loader : DataLoader
           The data loader.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Representations and labels respectively. Representations are in shape
            of N x cnn_dim x n_views shape while labels have shape of N.
        """

        # Save if model in training or eval mode
        was_training = self.model.training
        self.model.eval()

        # Lists storing embeddings and corresponding labels
        Z, Y = [], []

        # Define progress bar
        if self.verbose:
            title_suffix = 'Train' if data_loader == self.train_loader else 'Val'
            title = f'Generating embeddings | {title_suffix}'
            pbar = pkbar.Pbar(name=title, target=len(data_loader))

        Z = torch.zeros(len(data_loader.dataset), self.cnn_dim, self.n_views).half()
        Y = torch.zeros(len(data_loader.dataset)).long()
        next_i = 0

        # Generate embeddings
        with torch.no_grad():
            last_z = None
            last_y = None
            current_z = torch.zeros(self.batch_size, self.cnn_dim, self.n_views)
            current_z = current_z.half().to(self.device, non_blocking=True)

            for i, (x_views, y) in enumerate(self.iter_with_convert(data_loader, self.device)):

                # Move to CPU
                if last_z is not None:
                    last_z = AllGather.apply(last_z).to('cpu', non_blocking=True)
                    last_y = AllGather.apply(last_y).to('cpu', non_blocking=True)

                # Last batch size might be different
                current_z = current_z[:len(y)]

                # Save batch_size amount of embeddings of n views
                with torch.cuda.amp.autocast():
                    for j, x in enumerate(x_views):
                        current_z[:, :, j] = self.model(x)

                # Save previous embeddings
                if last_z is not None:
                    prev_i = next_i
                    next_i += len(last_z)
                    Z[prev_i:next_i] += last_z
                    Y[prev_i:next_i] += last_y
                    
                # Update previous embeddings with current
                last_z = current_z.clone()
                last_y = y

                # Step progress progress bar
                if self.verbose:
                    pbar.update(i)

        prev_i = next_i
        next_i += len(last_z)
        Z[prev_i:next_i] += last_z.cpu()
        Y[prev_i:next_i] += last_y.cpu()

        # Set back original mode of model, train or eval
        self.model.train(was_training)

        # Embeddings, labels
        return Z, Y
