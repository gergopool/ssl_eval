import torch
import torch.nn.functional as F
from typing import List, Union

from . import pkbar
from .utils import DistributedAverageMeter
from .data import create_lin_eval_dataloader
from .distributed import get_world_size_n_rank


class KNNEvaluator:
    """KNNEvaluator
    Calculates knn predictions and accuracies on given representations.
    Very important note that this is not exactly true knn, because
    every representation is normalized into a hyper sphere and the distances
    are then formed as cosine distances. This is done in order to get a faster
    evaluation.

    Parameters
    ----------
    device : torch.device
        The default device to use on freshly initialized tensors.
    verbose : bool, optional
        Whether to print results to standard output or not. Only rank0 process
        prints. By default True
    """

    def __init__(self, device: torch.device, verbose: bool = True):
        self.device = device
        self.world_size, self.rank = get_world_size_n_rank()
        self.verbose = verbose and self.rank == 0

    # =========================================================================
    # Public functions
    # =========================================================================

    def __call__(self,
                 train_z: torch.Tensor,
                 train_y: torch.Tensor,
                 val_z: torch.Tensor,
                 val_y: torch.Tensor,
                 k: Union[int, list]) -> torch.Tensor:
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
        k : Union[int, list]
            Desired K values to use for the algorithm.

        Returns
        -------
        torch.Tensor
            Tensor with length of K, accuracies in the order
            in which the k values were given.
        """

        # If needed, converts k from int to list of ints
        # The s stands from plural form of k (list)
        ks = self._correct_ks(k)
        largest_k = max(ks)

        # Use only first views
        train_z = train_z[..., 0]

        # Create extended labels
        train_y = train_y.repeat(500).view(500, -1)
        train_y = train_y.to(self.device)

        # Data loader
        data_loader = create_lin_eval_dataloader(val_z, val_y, batch_size=500)

        # Print
        if self.verbose:
            print("\nKNN-evaluation")
            pbar = pkbar.Kbar(target=len(data_loader))

        # Accuracy meter
        acc_meter = DistributedAverageMeter(self.device, n=len(ks))

        # Get knn prediction for each batch
        for z, y in data_loader:

            y = y.cpu()
            z = z.to(self.device)

            # This batch's accuracy (for logging purpose)
            batch_hits = torch.zeros(len(ks)).to(self.device)

            # Distance matrix
            dist = self._mm_splitwise_on_gpu(z, train_z).to(self.device)

            # Get closes labels
            closest_indices = dist.topk(largest_k, dim=1)[1]
            pred_labels = torch.gather(train_y, dim=1, index=closest_indices)

            # For each k get number of hits
            batch_hits = torch.zeros(len(ks)).to(self.device)
            for i, k in enumerate(ks):
                preds = pred_labels[:, :k].mode(dim=1)[0].cpu()
                batch_hits[i] = (preds == y).sum()
            acc_meter.update(batch_hits, n=len(y))

            # Log accuracy
            if self.verbose:
                accs = acc_meter.local_avg
                update_values = [(f"k={k}", acc) for (k, acc) in zip(ks, accs)]
                pbar.add(1, values=update_values)

        train_y = train_y.cpu()

        # Get all accuracies
        final_accuracies = acc_meter.avg

        # Log final, official values
        if self.verbose:
            for k, acc in zip(ks, final_accuracies):
                print(f"Top1 @ K={k:<2d} : {acc*100:3.2f}%")

        return final_accuracies

    # =========================================================================
    # Private functions
    # =========================================================================

    def _mm_splitwise_on_gpu(self, small: torch.Tensor, large: torch.Tensor) -> torch.Tensor:
        """_mm_splitwise_on_gpu
        Since imagenet's train representations require a large gpu memory, we cannot 
        calculate everything on the gpu at once. On the other hand it is neither desired
        to calcualte everything on CPU. Thus, we split the train embeddings into chunks
        and calculate the distances on the GPU in chunks iteratively.

        Parameters
        ----------
        small : torch.Tensor
            The smaller tensor which we can store in GPU in whole.
        large : torch.Tensor
            The larger tensor on which we iterate in chunks.

        Returns
        -------
        torch.Tensor
            _description_
        """

        # Normalize
        small = F.normalize(small.to(self.device), dim=1).half()

        # This is an arbitrary batch size that surely first onto
        # the gpu even if the representation's dimension is large (e.g. 8192)
        batch_size = 2500

        # The tensor in which we store the results
        results = torch.zeros(len(small), len(large)).to(self.device)

        # Iterate in chunks
        for start in range(0, len(large), batch_size):

            # Normalized chunk
            end = min(len(large), start + batch_size)
            z = F.normalize(large[start:end].to(self.device), dim=1)

            # Distances
            sub_result = small @ z.T

            # Store distances
            results[:, start:end] = sub_result

        return results

    def _correct_ks(self, ks: Union[list, int]) -> List[int]:
        """_correct_ks
        If Ks is not a list but an int, it creates a list out of it
        """

        if isinstance(ks, int):
            ks = [ks]

        if not isinstance(ks, list):
            raise TypeError(f"Value k must be either a list or int, not {type(ks)}")

        # Ensure all k values are int
        ks = [int(k) for k in ks]

        return ks