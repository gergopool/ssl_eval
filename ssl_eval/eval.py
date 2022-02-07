import torch
import torch.nn.functional as F
from torch import nn
from typing import Union, Tuple
import pkbar

from .distributed import AllGather, AllReduce, get_world_size_n_rank
from .data import get_loaders_by_name
from .utils import accuracy

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

    def generate_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:

        train_z, train_y = [], []

        if self.verbose:
            pbar = pkbar.Pbar(name='Generating embeddings', target=len(self.train_loader))

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                for i, (x, y) in enumerate(self.train_loader):
                    x = x.cuda()
                    y = y.cuda()
                    z = self.model(x)

                    train_z.append(z)
                    train_y.append(y)

                    if self.verbose:
                        pbar.update(i)

                train_z = AllGather.apply(torch.cat(train_z))
                train_y = AllGather.apply(torch.cat(train_y))

        return train_z, train_y

    def linear_eval(self,
                    train_z: torch.Tensor,
                    train_y: torch.Tensor,
                    epochs: int = 100,
                    batch_size: int = None,
                    lr: float = 1e-2) -> torch.Tensor:

        # Define linear classifier
        classifier = nn.Linear(self.cnn_dim, train_y.max() + 1).cuda()
        classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.bias.data.zero_()
        if self.world_size > 1:
            classifier = nn.parallel.DistributedDataParallel(classifier)

        # Optimizer and loss
        opt = torch.optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        # Choose large batch size for quick evaluation
        if batch_size is None:
            batch_size = 16384 if self.dataset == 'imagenet' else 512

        # Iterations per epoch
        ipe = len(train_z) // batch_size

        if self.verbose:
            # Progress bar
            pbar = pkbar.Pbar(f"\nLinear Eval - Training {epochs} epochs.", target=epochs)

        # Train N epochs
        for epoch in range(epochs):
            # Random permutation of data inidces for the batch
            indices = torch.randperm(len(train_z))

            for i in range(ipe):
                # Prepare data
                j = indices[i * batch_size:(i + 1) + batch_size]
                z = train_z[j].float()  # Ensure using float32 for high precision
                y = train_y[j]

                # Evaluate
                y_hat = classifier(z)
                loss = criterion(y_hat, y)

                # Backprop
                opt.zero_grad()
                loss.backward()
                opt.step()

            if self.verbose:
                pbar.update(epoch)

        if self.verbose:
            print(f"Linear Eval - Evaluating.")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        # Evaluate once
        accs = []
        for x, y in self.val_loader:
            # Prepare input x and y
            x = x.cuda()
            y = y.cuda()

            # Predict
            with torch.no_grad():
                y_hat = classifier(self.model(x))

            # Get accuracy
            acc = accuracy(y_hat, y)
            accs.append(acc)

            if self.verbose:
                pbar.add(1, values=[("acc", acc)])

        # Mean accuracy over all ranks
        acc = AllReduce.apply(torch.Tensor(accs).cuda()).mean()

        if self.verbose:
            print(f"Top1 @ Linear Eval: {acc*100:3.2f}%")

        return acc

    def _knn(self, train_z: torch.Tensor, train_y: torch.Tensor, ks: list) -> torch.Tensor:

        if isinstance(ks, int):
            ks = [ks]

        if not isinstance(ks, list):
            raise TypeError(f"Value k must be either a list or int, not {type(ks)}")

        # Ensure all k values are int
        ks = [int(k) for k in ks]

        # Normalize trained embeddings
        train_z = F.normalize(train_z, dim=1)

        # Prepare variables
        total = 0  # total number of data, length
        largest_k = max(ks)  # largest k value out of all
        n_hits = torch.zeros(len(ks)).cuda()  # Number of hits for each k
        train_y = train_y.repeat(self.batch_size).view(self.batch_size, -1)  # train labels

        if self.verbose:
            print("KNN-evaluation")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        # Get knn prediction for each batch
        for x, y in self.val_loader:
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    # Current batch's embeddings and labels
                    x = x.cuda()
                    y = y.cuda()
                    z = F.normalize(self.model(x))

                # This batch's accuracy (for logging purpose)
                batch_hits = torch.zeros(len(ks)).cuda()

                # Distance matrix
                dist = z @ train_z.T

                # Get closes labels
                closest_indices = dist.topk(largest_k, dim=1)[1]
                pred_labels = torch.gather(train_y, dim=1, index=closest_indices)

                # For each k get number of hits
                for j, k in enumerate(ks):
                    preds = pred_labels[:, :k].mode(dim=1)[0]
                    batch_hits[j] += (preds == y).sum()

            n_hits += batch_hits
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

        return accuracies


class KNNEvaluator:

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
        self.verbose = verbose and get_world_size_n_rank()[1] == 0

        data_loaders = get_loaders_by_name(root, dataset, batch_size=batch_size)
        self.train_loader, self.val_loader = data_loaders

    def _get_labeled_embeddings(self):

        features, labels = [], []

        if self.verbose:
            pbar = pkbar.Pbar(name='Generating embeddings', target=len(self.train_loader))

        for i, (x, y) in enumerate(self.train_loader):
            x = x.cuda()
            y = y.cuda()
            z = self.model(x)

            features.append(z)
            labels.append(y)

            if self.verbose:
                pbar.update(i)

        features = AllGather.apply(torch.cat(features))
        labels = AllGather.apply(torch.cat(labels))

        return features, labels

    def _lin_eval(self,
                  train_z: torch.Tensor,
                  train_y: torch.Tensor,
                  epochs: int = 100) -> torch.Tensor:
        classifier = nn.Linear(self.cnn_dim, train_y.max() + 1).cuda()
        classifier.weight.data.normal_(mean=0.0, std=0.01)
        classifier.bias.data.zero_()

        if get_world_size_n_rank()[0] > 1:
            classifier = nn.parallel.DistributedDataParallel(classifier)

        opt = torch.optim.SGD(classifier.parameters(), lr=1e-2, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        batch_size = 16384 if self.dataset == 'imagenet' else 512
        ipe = len(train_z) // batch_size

        if self.verbose:
            pbar = pkbar.Pbar(f"Linear Eval - Training {epochs} epochs.", target=epochs * ipe)

        for epoch in range(epochs):
            indices = torch.randperm(len(train_z))
            for i in range(ipe):
                start = i * batch_size
                end = start + batch_size
                j = indices[start:end]
                x = train_z[j].float()
                y = train_y[j]

                y_hat = classifier(x)
                loss = criterion(y_hat, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                if self.verbose:
                    pbar.update(epoch * ipe + i)

        if self.verbose:
            print(f"Linear Eval - Evaluating.")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        accs = []
        for x, y in self.val_loader:
            x = x.cuda()
            y = y.cuda()
            with torch.no_grad():
                y_hat = classifier(self.model(x))
            acc = accuracy(y_hat, y)
            accs.append(acc)
            if self.verbose:
                pbar.add(1, values=[("acc", acc)])

        acc = AllReduce.apply(torch.Tensor(accs).cuda()).mean()

        if self.verbose:
            print(f"Top1 @ Linear Eval: {acc*100:3.2f}%")

        return acc

    def _evaluate(self, train_z: torch.Tensor, train_y: torch.Tensor, ks: list) -> torch.Tensor:

        train_z = F.normalize(train_z, dim=1)
        ks = [int(k) for k in ks]
        largest_k = max(ks)
        results = torch.zeros(len(ks)).cuda()
        train_y = train_y.repeat(self.batch_size).view(self.batch_size, -1)

        if self.verbose:
            print("KNN-evaluation")
            pbar = pkbar.Kbar(target=len(self.val_loader))

        total = 0
        for x, y in self.val_loader:
            x = x.cuda()
            y = y.cuda()
            z = F.normalize(self.model(x))

            batch_results = torch.zeros(len(ks)).cuda()

            dist = z @ train_z.T
            closest_indices = dist.topk(largest_k, dim=1)[1]
            pred_labels = torch.gather(train_y, dim=1, index=closest_indices)
            for j, k in enumerate(ks):
                preds = pred_labels[:, :k].mode(dim=1)[0]
                hits = (preds == y).sum()
                batch_results[j] += hits

            results += batch_results
            total += len(y)

            if self.verbose:
                accs = list(batch_results.cpu().numpy() * 100 / len(y))
                update_values = [(f"k={k}", acc) for (k, acc) in zip(ks, accs)]
                pbar.add(1, values=update_values)

        results /= total
        results = AllReduce.apply(results)

        if self.verbose:
            accs = list(results.cpu().numpy())
            for k, acc in zip(ks, accs):
                print(f"Top1 @ K={k:<2d} : {acc*100:3.2f}%")

        return results

    def __call__(self, ks: Union[int, list] = 1) -> torch.Tensor:

        if isinstance(ks, int):
            ks = [ks]

        if not isinstance(ks, list):
            raise TypeError(f"Value k must be either a list or int, not {type(ks)}")

        # Freee batchnorm
        training_mode_used_before = self.model.training
        self.model.eval()

        with torch.cuda.amp.autocast(enabled=True):
            with torch.no_grad():
                train_features, train_labels = self._get_labeled_embeddings()
                results = self._evaluate(train_features, train_labels, ks)

        self._lin_eval(train_features, train_labels)

        self.model.train(training_mode_used_before)

        return results
