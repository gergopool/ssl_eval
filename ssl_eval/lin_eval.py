import torch
from torch import nn
from copy import deepcopy
from torch.utils import data
from typing import Tuple
from torch.utils.data import DataLoader

from . import pkbar
from .distributed import get_world_size_n_rank
from .larc import LARC
from .early_stopping import EarlyStopping
from .data import create_lin_eval_dataloader
from .utils import DistributedAverageMeter, iter_with_convert

__all__ = ["LinearEvaluator"]

# The stored best model for warm start
_best_model = None


class LinearEvaluator:
    """LinearEvaluator

    Runs a linear evaluation on pre-generated embeddings.
    It does not provide the exact best performance of your network, but
    it can approximate its performance very well much faster.

    Parameters
    ----------
    cnn_dim : int
        Output dimension of encoder
    n_classes : int
        Number of classes
    device : torch.device
        Device on which the linear classifier network is defined.
    verbose : bool, optional
        Whether to print results to standard output or not. Only rank0 process
        prints. By default True
    warm_start : bool, optional
        If True and there has been a previous training, it loads the weights
        from the last training. By default True
    """

    def __init__(self,
                 cnn_dim: int,
                 n_classes: int,
                 device: torch.device,
                 verbose: bool = True,
                 warm_start: bool = False):
        self.cnn_dim = cnn_dim
        self.n_classes = n_classes
        self.device = device
        self.world_size, self.rank = get_world_size_n_rank()
        self.verbose = verbose and self.rank == 0
        self.warm_start = warm_start

        self.classifier = self._create_classifier()

    def _create_classifier(self) -> nn.Module:
        """Creates linear classifier module, distributed, if needed"""
        model = LinClassifier(self.cnn_dim, self.n_classes).to(self.device)
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[self.device],
                                                        output_device=self.device)
        return model

    def _load_best_classifier(self):
        """Loads in saved best model"""
        global _best_model
        if _best_model is not None:
            self.classifier = deepcopy(_best_model)

    def _save_best_classifier(self):
        """Saves current model as best"""
        global _best_model
        _best_model = deepcopy(self.classifier)

    def __call__(self,
                 train_z: torch.Tensor,
                 train_y: torch.Tensor,
                 val_z: torch.Tensor,
                 val_y: torch.Tensor,
                 epochs: int = 100,
                 batch_size: int = 256,
                 lr: float = 0.1) -> torch.Tensor:
        """__call__
        Runs linear evaluation on data.

        Parameters
        ----------
        train_z : torch.Tensor
            Train embeddings
        train_y : torch.Tensor
            Train labels
        val_z : torch.Tensor
            Validation embeddings on which we evaluate
        val_y : torch.Tensor
            Labels to validation embeddings
        epochs : int, optional
           Number of epochs, by default 100
        batch_size : int, optional
            Batch size, by default 256
        lr : float, optional
            Learning rate for LARC, by default 0.1

        Returns
        -------
        torch.Tensor
            Top1 accuracy.
        """
        train_loader = create_lin_eval_dataloader(train_z, train_y, batch_size)
        val_loader = create_lin_eval_dataloader(val_z, val_y, batch_size)
        self._train(train_loader, val_loader, epochs, batch_size, lr)
        accuracy = self._test(val_loader)
        return accuracy

    def _train(self,
               train_loader: data.DataLoader,
               val_loader: data.DataLoader,
               epochs: int = 100,
               batch_size: int = 256,
               lr: float = 0.1) -> nn.Module:
        """_train
        Trains a linear classifier while periodically validating
        it. While validation might seem cheating, this only helps
        to get a better approximation of what an official linear
        evaluation might achieve.

        Parameters
        ----------
        train_loader : data.DataLoader
            Data loader to train set
        val_loader : data.DataLoader
            Data loader to validation set
        epochs : int, optional
           Number of epochs, by default 100
        batch_size : int, optional
            Batch size, by default 256
        lr : float, optional
            Learning rate for LARC, by default 0.1

        Returns
        -------
        nn.Module
            The best classifier found.
        """

        if self.warm_start:
            self._load_best_classifier()

        # Optimizer
        opt = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, min_lr=lr / 100.)
        opt = LARC(opt, trust_coefficient=0.001, clip=False)

        # Stop early if classifier seems to overfit
        early_stopper = EarlyStopping(patience=15)

        # Log train start
        if self.verbose:
            bs = batch_size * self.world_size
            print(f"\nLinear Eval - params: lr={lr:0.5f} | batch_size={bs} | " +
                  f"num_GPUs: {self.world_size}")
            print(f"Linear Eval - Training {epochs} epochs.")
            pbar = pkbar.Kbar(target=epochs)  # Progress bar

        for epoch in range(epochs):

            # Set different shuffle
            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            # Train 1 epoch
            self.classifier.train()
            train_loss, train_acc = self._run_epoch(train_loader, opt=opt)

            # Validate
            self.classifier.eval()
            val_loss, val_acc = self._run_epoch(val_loader)

            # Decrease learing rate if val los not improving
            scheduler.step(val_loss)

            # Check if model not improved for a long time
            should_stop = early_stopper(val_loss, self.classifier)

            # Log this epoch
            if self.verbose:
                pbar_state = epochs if should_stop else epoch + 1
                pbar.update(pbar_state,
                            values=[('train_loss', train_loss), ("train_acc", train_acc),
                                    ('val_loss', val_loss), ("val_acc", val_acc)])
            # Quit if needed
            if should_stop:
                break

        # Restore and save best model
        self.classifier = early_stopper.best_model
        self._save_best_classifier()

        return self.classifier

    def _test(self, val_loader: data.DataLoader) -> torch.Tensor:
        """_test
        One epoch of official validation.

        Parameters
        ----------
        val_loader : data.DataLoader
            The validation data loader.

        Returns
        -------
        torch.Tensor
            Accuracy
        """
        self.classifier.eval()
        _, val_acc = self._run_epoch(val_loader)
        if self.verbose:
            acc_value = val_acc.cpu().numpy()
            print(f"Top1 @ Linear Eval: {acc_value*100:3.2f}%")
        return val_acc

    def _run_epoch(self,
                   data_laoder: data.DataLoader,
                   opt: torch.optim.Optimizer = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """_run_epoch
        Runs a single epoch on a data loader.

        Parameters
        ----------
        data_laoder : data.DataLoader
            The dataloader, either train or val.
        opt : torch.optim.Optimizer, optional
            Optimizer. If given, backpropagation applied. By default None

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Loss and accuracy, respectively.
        """

        # Trackers
        loss_meter = DistributedAverageMeter(self.device)
        acc_meter = DistributedAverageMeter(self.device)

        # CE loss
        criterion = nn.CrossEntropyLoss().to(self.device, non_blocking=True)

        for z, y in iter_with_convert(data_laoder, self.device):

            # Calculate y_hat
            y_hat = self.classifier(z.to(torch.float32, non_blocking=True))

            # Loss
            loss = criterion(y_hat, y)

            # Accuracy
            n_hits = (y_hat.argmax(dim=1) == y).sum()
            loss_meter.update(loss.item(), n=1)
            acc_meter.update(n_hits, len(y))

            # Backprop
            if opt is not None:
                opt.zero_grad()
                loss.backward()
                opt.step()

        return loss_meter.avg[0], acc_meter.avg[0]


class LinClassifier(nn.Module):
    """LinClassifier 
    Linear module trained on the top of the encoder.
    This is not exactly the official set up, because it is extended
    with a batch norm layer to speed up the convergence.

    Parameters
    ----------
    cnn_dim : int
        Output dimension of encoder
    n_classes : int
        Number of classes.
    """

    def __init__(self, cnn_dim: int, n_classes: int):
        super(LinClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(cnn_dim)
        self.fc = nn.Linear(cnn_dim, n_classes)
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        return self.fc(self.bn(x))