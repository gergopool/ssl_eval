from random import sample
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Union

from . import pkbar
from .utils import DistributedAverageMeter
from .data import create_lin_eval_dataloader
from .distributed import get_world_size_n_rank

from .knn import KNNEvaluator


class SNNEvaluator(KNNEvaluator):
    """SNNEvaluator
    Calculates snn predictions and accuracies on given representations.

    Parameters
    ----------
    tau : float
        The similarity weight before softmax. By default 0.07
    device : torch.device
        The default device to use on freshly initialized tensors.
    verbose : bool, optional
        Whether to print results to standard output or not. Only rank0 process
        prints. By default True
    """

    def __init__(self, *args, tau: float = 0.07, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.softmax = nn.Softmax(dim=1)

    # =========================================================================
    # Public functions
    # =========================================================================

    def __call__(self,
                 train_z: torch.Tensor,
                 train_y: torch.Tensor,
                 val_z: torch.Tensor,
                 val_y: torch.Tensor,
                 single_view: bool = True,
                 balance_labels: bool = False) -> torch.Tensor:
        """__call__
        Calculating knn accuracies.

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
        single_view : bool
            Use the first view only from the train set. Otherwise the different views
            act as different data points
        balance_labels : bool
            If the train data is not balanced, balance the weights of votes.

        Returns
        -------
        torch.Tensor
            Accuracy
        """

        # Use only first views
        # if single_view:
        #     train_z = train_z[..., 0].unsqueeze(-1)

        train_y = train_y.to(self.device)
        train_z = train_z[..., 0]

        # Weights of labels
        n_points = len(train_y)
        sample_weights = torch.ones(n_points)
        if balance_labels:
            label_weights = 1 / (torch.bincount(train_y) + 1e-8)
            sample_weights = label_weights[train_y]
        sample_weights = sample_weights.to(self.device)

        # # Create extended labels
        # train_y = train_y.repeat(500).view(500, -1)

        # Data loader
        data_loader = create_lin_eval_dataloader(val_z, val_y, batch_size=500)

        # Print
        if self.verbose:
            print("\nSNN-evaluation")
            pbar = pkbar.Kbar(target=len(data_loader))

        # Accuracy meter
        acc_meter = DistributedAverageMeter(self.device)

        # Get knn prediction for each batch
        for z, y in data_loader:

            y = y.to(self.device)
            z = z.to(self.device)

            # This batch's accuracy (for logging purpose)
            batch_hits = 0

            # Distance matrix
            dist = self._mm_splitwise_on_gpu(z, train_z)  # len(z) x len(train_z)

            # Probabilities
            batch_probabilities = self.softmax(dist / self.tau)

            for i, probabilities in enumerate(batch_probabilities):
                pred = torch.bincount(train_y, weights=probabilities * sample_weights).argmax()
                batch_hits += int(pred == y[i])

            acc_meter.update(batch_hits, n=len(y))

            # Log accuracy
            if self.verbose:
                update_values = [(f"acc", acc_meter.local_avg)]
                pbar.add(1, values=update_values)

        # Get all accuracies
        final_accuracy = acc_meter.avg

        # Set back to cpu
        train_y = train_y.cpu()

        # Log final, official values
        if self.verbose:
            print(f"Top1 @ SNN : {float(final_accuracy)*100:3.2f}%")

        return final_accuracy