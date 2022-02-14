import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from typing import Tuple
import pkbar

from .distributed import AllGather, AllReduce, get_world_size_n_rank
from .data import get_loaders_by_name, create_lin_eval_dataloader

__all__ = ['Evaluator']


class Evaluator:

    def __init__(self,
                 model: nn.Module,
                 cnn_dim: int,
                 dataset: str,
                 root: str,
                 batch_size=256,
                 verbose=True):
        self.model = model
        self.cnn_dim = cnn_dim
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size, self.rank = get_world_size_n_rank()
        self.verbose = verbose and self.rank == 0

        data_loaders = get_loaders_by_name(root, dataset, batch_size=batch_size)
        self.train_loader, self.val_loader = data_loaders

    def generate_embeddings(self, flip=False) -> Tuple[torch.Tensor, torch.Tensor]:

        was_training = self.model.training
        self.model.eval()

        train_z, train_y = [], []

        if self.verbose:
            pbar = pkbar.Pbar(name='\nGenerating embeddings', target=len(self.train_loader))

        with torch.no_grad():
            for i, (x, y) in enumerate(self.train_loader):
                x = x.cuda()
                y = y.cuda()
                if flip:
                    flipped_x = torchvision.transforms.functional.hflip(x)
                    x = torch.cat((x, flipped_x))
                z = self.model(x)
                assert z.shape[-1] == self.cnn_dim, f"Expected dim {self.cnn_dim} but got dim {z.shape[-1]}"
                if flip:
                    z = z.view(2, -1, self.cnn_dim).permute(1, 2, 0)
                z = AllGather.apply(z).cpu()
                y = AllGather.apply(y).cpu()

                train_z.append(z)
                train_y.append(y)

                if self.verbose:
                    pbar.update(i)

            train_z = torch.cat(train_z)
            train_y = torch.cat(train_y)

        self.model.train(was_training)

        return train_z.float(), train_y

    def _get_start_weights(self, train_z, train_y):
        n_classes = int(train_y.max() + 1)
        if train_z.dim() == 3 and train_z.shape[-1] == 2:
            train_z = train_z.mean(dim=-1)
        z = F.normalize(train_z, dim=1).float()
        weights = torch.zeros(n_classes, train_z.shape[-1]).cuda()
        for y in range(n_classes):
            i = (train_y == y).nonzero().ravel()
            mean_vec = z[i].mean(dim=0)
            weights[y] = F.normalize(mean_vec, dim=0)

        scale = torch.norm(train_z, dim=1).mean() / 10
        return weights / scale

    def adjust_learning_rate(self, optimizer, init_lr, epoch, epochs):
        """Decay the learning rate based on schedule"""
        import math
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
        for param_group in optimizer.param_groups:
            param_group['lr'] = cur_lr

    def linear_eval(self,
                    train_z: torch.Tensor,
                    train_y: torch.Tensor,
                    epochs: int = 10,
                    batch_size: int = 256,
                    lr: float = 1.) -> torch.Tensor:

        # Define linear classifier
        n_classes = int(train_y.max() + 1)
        classifier = nn.Linear(self.cnn_dim, n_classes).cuda()
        # classifier.weight.data.copy_(self._get_start_weights(train_z, train_y))
        classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.bias.data.zero_()
        if self.world_size > 1:
            classifier = nn.parallel.DistributedDataParallel(classifier)

        # Remember original trianing mode
        was_training = self.model.training
        self.model.eval()

        # Speed up running
        torch.backends.cudnn.benchmark = True

        # Optimizer and loss
        opt = torch.optim.Adam(classifier.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss().cuda()

        # Dataloader, distributed if world_size > 1
        data_loader = create_lin_eval_dataloader(train_z, train_y, batch_size)

        if self.verbose:
            print(f"\nLinear Eval - params: lr={lr:0.5f} | batch_size={batch_size}")
            print(f"Linear Eval - Training {epochs} epochs.")
            pbar = pkbar.Kbar(target=epochs * len(data_loader))  # Progress bar

        # Train N epochs
        for epoch in range(epochs):
            # Random permutation of data inidces for the batch
            hits = 0
            total = 0

            self.adjust_learning_rate(opt, lr, epoch, epochs)

            if self.world_size > 1:
                data_loader.sampler.set_epoch(epoch)

            for z, y in data_loader:
                opt.zero_grad()
                # Prepare data
                z = z.cuda()
                y = y.cuda()

                # Evaluate
                y_hat = classifier(z)
                loss = criterion(y_hat, y)

                # Acc data
                hits += (y_hat.argmax(dim=1) == y).sum()
                total += len(y)

                # Backprop
                loss.backward()
                opt.step()

                if self.verbose:
                    acc = (y_hat.argmax(dim=1) == y).sum() / len(y)
                    pbar.add(1, values=[('loss', loss.item()), ("acc", acc)])

        if self.verbose:
            print(f"Linear Eval - Evaluating.")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        # Evaluate once
        hits = torch.zeros(1).cuda()
        total = torch.zeros(1).cuda()
        for x, y in self.val_loader:
            # Prepare input x and y
            x = x.cuda()
            y = y.cuda()

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
        small = small.cuda()
        batch_size = 10000
        results = torch.zeros(len(small), len(large)).cuda()
        for start in range(0, len(large), batch_size):
            end = min(len(large), start + batch_size)
            z = large[start:end].cuda()
            sub_result = small @ z.T
            results[:, start:end] = sub_result
        return results.cpu()

    def knn(self, train_z: torch.Tensor, train_y: torch.Tensor, ks: list) -> torch.Tensor:

        if isinstance(ks, int):
            ks = [ks]

        if not isinstance(ks, list):
            raise TypeError(f"Value k must be either a list or int, not {type(ks)}")

        # Ensure all k values are int
        ks = [int(k) for k in ks]

        device = train_z.device

        was_training = self.model.training
        self.model.eval()

        # Normalize trained embeddings
        train_z = F.normalize(train_z, dim=1)

        # Ignore flip mode, use only the original training image
        if train_z.dim() == 3 and train_z.shape[-1] == 2:
            train_z = train_z[..., 0]

        # Prepare variables
        total = 0  # total number of data, length
        largest_k = max(ks)  # largest k value out of all
        n_hits = torch.zeros(len(ks)).cuda()  # Number of hits for each k
        train_y = train_y.repeat(self.batch_size).view(self.batch_size, -1)  # train labels

        if self.verbose:
            print("\nKNN-evaluation")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        # Get knn prediction for each batch
        for x, y in self.val_loader:
            with torch.no_grad():
                # Current batch's embeddings and labels
                x = x.cuda()
                y = y.to(device)
                z = F.normalize(self.model(x)).to(device)

            # This batch's accuracy (for logging purpose)
            batch_hits = torch.zeros(len(ks)).to(device)

            # Distance matrix
            dist = self._mm_splitwise_on_gpu(z, train_z)

            # Get closes labels
            closest_indices = dist.topk(largest_k, dim=1)[1]
            pred_labels = torch.gather(train_y, dim=1, index=closest_indices)

            # For each k get number of hits
            for j, k in enumerate(ks):
                preds = pred_labels[:, :k].mode(dim=1)[0]
                batch_hits[j] += (preds == y).sum()

            n_hits += batch_hits.cuda()
            total += len(y)

            if self.verbose:
                # Log accuracy
                accs = list(batch_hits.cpu().numpy() * 100 / len(y))
                update_values = [(f"k={k}", acc) for (k, acc) in zip(ks, accs)]
                pbar.add(1, values=update_values)

        # Get accuracy tensor for each k
        accuracies = AllReduce.apply(n_hits / total)

        if self.verbose:
            accuracy_list = list(accuracies.cpu().numpy())
            for k, acc in zip(ks, accuracy_list):
                print(f"Top1 @ K={k:<2d} : {acc*100:3.2f}%")

        self.model.train(was_training)

        return accuracies