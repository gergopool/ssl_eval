import torch
import torch.nn.functional as F
from torch import nn
from typing import Tuple
import pkbar
from collections import OrderedDict
from functools import cached_property

from .distributed import AllGather, AllReduce, get_world_size_n_rank
from .data import get_loaders_by_name, create_lin_eval_dataloader
from .early_stopping import EarlyStopping
from .larc import LARC

__all__ = ['Evaluator']


class Evaluator:

    def __init__(self, model: nn.Module, dataset: str, root: str, batch_size=256, verbose=True):
        self.model = model
        self.dataset = dataset
        self.root = root
        self.batch_size = batch_size
        self.world_size, self.rank = get_world_size_n_rank()
        self.verbose = verbose and self.rank == 0

        self.reset_data_loaders()

    @property
    def device(self):
        return next(self.model.parameters()).device

    @cached_property
    def cnn_dim(self):
        shape = (3, 244, 244) if self.dataset == 'imagenet' else (3, 32, 32)
        fake_input = torch.zeros(1, *shape).to(self.device)
        x = self.model(fake_input)
        return len(x[0])

    def reset_data_loaders(self, n_views: int = 1):

        data_loaders = get_loaders_by_name(self.root,
                                           self.dataset,
                                           batch_size=self.batch_size,
                                           n_views=n_views)
        self.train_loader, self.val_loader = data_loaders

    def generate_embeddings(self, n_views: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:

        self.reset_data_loaders(n_views)

        was_training = self.model.training
        self.model.eval()

        train_z, train_y, val_z, val_y = [], [], [], []

        if self.verbose:
            length = len(self.train_loader) + len(self.val_loader)
            pbar = pkbar.Pbar(name='\nGenerating embeddings', target=length)

        with torch.no_grad():

            # TRAIN
            for i, (x_views, y) in enumerate(self.train_loader):
                y = y.to(self.device)

                # Save batch_size amount of embeddings of n views
                z = torch.zeros(len(y), self.cnn_dim, n_views).to(self.device)
                with torch.cuda.amp.autocast():
                    for j, x in enumerate(x_views):
                        z[:, :, j] = self.model(x.to(self.device))

                z = AllGather.apply(z).float().cpu()
                y = AllGather.apply(y).cpu()

                train_z.append(z)
                train_y.append(y)

                if self.verbose:
                    pbar.update(i)

            # VAL
            for i, (x, y) in enumerate(self.val_loader):
                y = y.to(self.device)
                with torch.cuda.amp.autocast():
                    z = self.model(x.to(self.device))
                z = z.unsqueeze(-1).float()
                z = AllGather.apply(z).cpu()
                y = AllGather.apply(y).cpu()

                val_z.append(z)
                val_y.append(y)

                if self.verbose:
                    pbar.update(len(train_y) + i)

            train_z = torch.cat(train_z)
            train_y = torch.cat(train_y)
            val_z = torch.cat(val_z)
            val_y = torch.cat(val_y)

        self.model.train(was_training)

        return train_z, train_y, val_z, val_y

    def linear_eval(self,
                    train_z: torch.Tensor,
                    train_y: torch.Tensor,
                    val_z: torch.Tensor,
                    val_y: torch.Tensor,
                    epochs: int = 100,
                    batch_size: int = 256,
                    lr: float = 0.1,
                    scale_lr=True) -> torch.Tensor:

        # Define linear classifier
        n_classes = int(val_y.max() + 1)
        classifier = nn.Sequential(
            OrderedDict([('bn', nn.BatchNorm1d(self.cnn_dim, affine=False)),
                         ('fc', nn.Linear(self.cnn_dim, n_classes))])).to(self.device)

        classifier.fc.weight.data.normal_(mean=0.0, std=0.01)
        classifier.fc.bias.data.zero_()
        if self.world_size > 1:
            classifier = nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
            classifier = nn.parallel.DistributedDataParallel(classifier,
                                                             device_ids=[self.device],
                                                             output_device=self.device)

        if scale_lr:
            lr = self.world_size * batch_size / 256 * lr

        # Remember original trianing mode
        was_training = self.model.training
        self.model.eval()

        # Speed up running
        torch.backends.cudnn.benchmark = True

        # Optimizer and loss
        opt = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, min_lr=lr / 100.)
        opt = LARC(opt, trust_coefficient=0.001, clip=False)
        criterion = nn.CrossEntropyLoss().to(self.device)
        early_stopper = EarlyStopping(patience=15, restore_best=True)

        # Dataloader, distributed if world_size > 1
        train_loader = create_lin_eval_dataloader(train_z, train_y, batch_size)
        val_loader = create_lin_eval_dataloader(val_z, val_y, batch_size)

        if self.verbose:
            bs = batch_size * self.world_size
            print(f"\nLinear Eval - params: lr={lr:0.5f} | batch_size={bs} | " +
                  f"num_GPUs: {self.world_size}")
            print(f"Linear Eval - Training {epochs} epochs.")
            pbar = pkbar.Kbar(target=epochs)  # Progress bar

        # Train N epochs
        for epoch in range(epochs):
            # Random permutation of data inidces for the batch

            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

            train_loss = torch.zeros(1).to(self.device)
            train_hits = torch.zeros(1).to(self.device)
            train_total = torch.zeros(1).to(self.device)
            val_loss = torch.zeros(1).to(self.device)
            val_hits = torch.zeros(1).to(self.device)
            val_total = torch.zeros(1).to(self.device)

            classifier.train()
            for z, y in train_loader:
                opt.zero_grad()
                # Prepare data
                z = z.to(self.device)
                y = y.to(self.device)

                # Evaluate
                y_hat = classifier(z)
                loss = criterion(y_hat, y)

                train_hits += (y_hat.argmax(dim=1) == y).sum()
                train_total += len(y)
                train_loss += loss

                # Backprop
                loss.backward()
                opt.step()

            classifier.eval()

            for z, y in val_loader:
                z = z.to(self.device)
                y = y.to(self.device)

                with torch.no_grad():
                    y_hat = classifier(z)

                val_hits += (y_hat.argmax(dim=1) == y).sum()
                val_total += len(y)

                loss = criterion(y_hat, y)
                val_loss += loss

            train_hits = AllGather.apply(train_hits).sum()
            train_total = AllGather.apply(train_total).sum()
            train_loss = AllReduce.apply(train_loss / len(train_loader))
            train_acc = train_hits / train_total

            val_hits = AllGather.apply(val_hits).sum()
            val_total = AllGather.apply(val_total).sum()
            val_loss = AllReduce.apply(val_loss / len(val_loader))
            val_acc = val_hits / val_total

            scheduler.step(val_loss)
            should_stop = early_stopper(val_loss, classifier)

            if self.verbose:
                pbar_state = epochs if should_stop else epoch
                pbar.update(pbar_state,
                            values=[('train_loss', train_loss), ("train_acc", train_acc),
                                    ('val_loss', val_loss), ("val_acc", val_acc)])

            if should_stop:
                break

        if self.verbose:
            print(f"Linear Eval - Evaluating.")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        classifier = early_stopper.best_model

        # Evaluate once
        classifier.eval()
        hits = torch.zeros(1).to(self.device)
        total = torch.zeros(1).to(self.device)
        for x, y in self.val_loader:
            # Prepare input x and y
            x = x.to(self.device)
            y = y.to(self.device)

            # Predict
            with torch.no_grad():
                y_hat = classifier(self.model(x))

            # Get accuracy
            batch_hits = (y_hat.argmax(1) == y).sum()
            hits += batch_hits
            total += len(y)

            if self.verbose:
                pbar.add(1, values=[("acc", batch_hits / len(y))])

        # Mean accuracy over all ranks
        hits = AllGather.apply(hits).sum()
        total = AllGather.apply(total).sum()
        acc = hits / total

        if self.verbose:
            acc_value = acc.cpu().numpy()
            print(f"Top1 @ Linear Eval: {acc_value*100:3.2f}%")

        self.model.train(was_training)

        return acc

    def _mm_splitwise_on_gpu(self, small, large):
        small = small.to(self.device)
        batch_size = 10000
        results = torch.zeros(len(small), len(large)).to(self.device)
        for start in range(0, len(large), batch_size):
            end = min(len(large), start + batch_size)
            z = large[start:end].to(self.device)
            sub_result = small @ z.T
            results[:, start:end] = sub_result
        return results.cpu()

    def knn(self,
            train_z: torch.Tensor,
            train_y: torch.Tensor,
            val_z: torch.Tensor,
            val_y: torch.Tensor,
            ks: list) -> torch.Tensor:

        if isinstance(ks, int):
            ks = [ks]

        if not isinstance(ks, list):
            raise TypeError(f"Value k must be either a list or int, not {type(ks)}")

        # Ensure all k values are int
        ks = [int(k) for k in ks]

        was_training = self.model.training
        self.model.eval()

        # Normalize trained embeddings
        train_z = F.normalize(train_z, dim=1)

        # Take only 1 view for this
        train_z = train_z[..., 0]

        # Prepare variables
        total = torch.zeros(1).to(self.device)  # total number of data, length
        largest_k = max(ks)  # largest k value out of all
        n_hits = torch.zeros(1, len(ks)).to(self.device)  # Number of hits for each k
        train_y = train_y.repeat(500).view(500, -1)  # train labels

        val_loader = create_lin_eval_dataloader(val_z, val_y, batch_size=500)

        if self.verbose:
            print("\nKNN-evaluation")
            pbar = pkbar.Kbar(target=len(val_loader))

        # Get knn prediction for each batch
        for z, y in val_loader:

            with torch.no_grad():
                # Current batch's embeddings and labels
                y = y.cpu()
                z = F.normalize(z.to(self.device))

            # This batch's accuracy (for logging purpose)
            batch_hits = torch.zeros(len(ks)).to(self.device)

            # Distance matrix
            dist = self._mm_splitwise_on_gpu(z, train_z)

            # Get closes labels
            closest_indices = dist.topk(largest_k, dim=1)[1]
            pred_labels = torch.gather(train_y, dim=1, index=closest_indices)

            # For each k get number of hits
            for j, k in enumerate(ks):
                preds = pred_labels[:, :k].mode(dim=1)[0]
                batch_hits[j] += (preds == y).sum()

            n_hits[0] += batch_hits
            total += len(y)

            if self.verbose:
                # Log accuracy
                accs = list(batch_hits.cpu().numpy() * 100 / len(y))
                update_values = [(f"k={k}", acc) for (k, acc) in zip(ks, accs)]
                pbar.add(1, values=update_values)

        # Get accuracy tensor for each k
        n_hits = AllGather.apply(n_hits).sum(dim=0)
        total = AllGather.apply(total).sum()
        accuracies = n_hits / total

        if self.verbose:
            accuracy_list = list(accuracies.cpu().numpy())
            for k, acc in zip(ks, accuracy_list):
                print(f"Top1 @ K={k:<2d} : {acc*100:3.2f}%")

        self.model.train(was_training)

        return accuracies