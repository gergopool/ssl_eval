import torch
import pkbar
from functools import cached_property

from .data import get_loaders_by_name
from .distributed import AllGather


class EmbGenerator:

    def __init__(self,
                 model,
                 dataset,
                 root,
                 n_views=1,
                 batch_size=256,
                 verbose=True):
        self._model = model
        self._dataset = dataset
        self._root = root
        self._verbose = verbose
        self._n_views = n_views
        self._batch_size = batch_size

        data_loaders = get_loaders_by_name(self._root,
                                           self._dataset,
                                           batch_size=self.batch_size,
                                           n_views=n_views)
        self.train_loader, self.val_loader = data_loaders

    # @cached_property
    # def cnn_dim(self):
    #     shape = (3, 244, 244) if self._dataset == 'imagenet' else (3, 32, 32)
    #     fake_input = torch.zeros(1, *shape).to(self.device)
    #     x = self._model(fake_input)
    #     return len(x[0])

    def get_val_embs(self):
        return self._generate(self.val_loader)

    def get_train_embs(self):
        return self._generate(self.train_loader)

    def _generate(self, data_loader):

        # Save if model in training or eval mode
        was_training = self._model.training
        self._model.eval()

        # Lists storing embeddings and corresponding labels
        Z, Y = [], []

        # Define progress bar
        if self._verbose:
            title_suffix = 'Train' if data_loader == self.train_loader else 'Val'
            title = f'Generating embeddings | {title_suffix}'
            pbar = pkbar.Pbar(name=title, target=len(data_loader))

        # Generate embeddings
        for i, (x_views, y) in enumerate(data_loader):

            # Move labels to GPU in order to be gatherable
            y = y.to(self.device)

            # Save batch_size amount of embeddings of n views
            # z.shape == batch_size x cnn_dim x n_views
            z = torch.zeros(len(y), -1, self._n_views).to(self.device)
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    for j, x in enumerate(x_views):
                        z[:, :, j] = self._model(x.to(self.device))

            # Collect embeddings from all GPU and save to CPU
            Z.append(AllGather.apply(z).float().cpu())
            Y.append(AllGather.apply(y).cpu())

            # Step progress progress bar
            if self._verbose:
                pbar.update(i)

        # Set back original mode of model, train or eval
        self._model.train(was_training)

        # Embeddings, labels
        return torch.cat(Z), torch.cat(Y)
