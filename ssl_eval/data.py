import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from typing import List, Tuple, Callable

from .distributed import get_world_size_n_rank

__all__ = ['get_loaders_by_name', "create_lin_eval_dataloader"]

# ===================================================================
# Public functions
# ===================================================================


def get_loaders_by_name(root: str, dataset_name: str, **kwargs) -> Tuple[DataLoader, DataLoader]:
    """get_loaders_by_name
    Given the name of the dataset and its path creates the
    train and validation dataloaders.

    Parameters
    ----------
    root : str
        The path to the dataset's root.
    dataset_name : str
        Name of the dataset. Choose from 'imagenet', 'tiny_imagenet', 'cifar10' and 'cifar100'.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Train and validation data loaders, respectively.

    Raises
    ------
    NameError
        If dataset name is not found.
    """
    if dataset_name in ['imagenet', 'tiny_imagenet', 'cifar10', 'cifar100']:
        return globals()[f"_{dataset_name}"](root, **kwargs)
    else:
        raise NameError(
            f"Unknown dataset name: {dataset_name}. " +\
             "Please choose from [imagenet, tiny_imagenet, cifar10, cifar100]"
        )


def create_lin_eval_dataloader(z: torch.Tensor,
                               y: torch.Tensor,
                               batch_size: int = 1000) -> DataLoader:
    """create_lin_eval_dataloader
    Given a list of embeddings in a form of a 3d torch tensor, and given
    their corresponding labels in an 1d tensor it creates a torch dataloader
    which both speeds up iterating over the dataset and manages multi-gpu
    access.

    Parameters
    ----------
    z : torch.Tensor
        3D tensor in shape (batch_size x cnn_dim x n_views).
    y : torch.Tensor
        1D tensor, the labels.
    batch_size : int, optional
        Batch size of the data loader. By default 1000

    Returns
    -------
    DataLoader
        The torchvision DataLoader.
    """

    emb_dataset = _EmbeddingDataset(z, y)

    # Get world size and current rank of this process
    world_size, rank = get_world_size_n_rank()

    # Convert to distributed dataset if required
    if world_size > 1:
        sampler = DistributedSampler(emb_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    # Create data loaders
    data_loader = DataLoader(emb_dataset,
                             num_workers=8,
                             pin_memory=True,
                             persistent_workers=True,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             sampler=sampler)

    return data_loader


# ===================================================================
# Private functions
# ===================================================================


def _get_loaders(train_dataset: Dataset,
                 val_dataset: Dataset,
                 batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:
    """_get_loaders
    Creates single or distributed data loaders out from the datasets.

    Parameters
    ----------
    train_dataset : Dataset
        Train dataset
    val_dataset : Dataset
        Validation dataset
    batch_size : int, optional
        Batch size of data loader, by default 256

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        Train and validation data loaders, respectively.
    """

    # Get world size and current rank of this process
    world_size, rank = get_world_size_n_rank()

    # Convert to distributed dataset if required
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset,
                                           num_replicas=world_size,
                                           rank=rank,
                                           shuffle=True)
        val_sampler = DistributedSampler(val_dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              num_workers=8,
                              pin_memory=True,
                              persistent_workers=True,
                              batch_size=batch_size,
                              sampler=train_sampler)
    val_loader = DataLoader(val_dataset,
                            num_workers=8,
                            pin_memory=True,
                            persistent_workers=True,
                            batch_size=batch_size,
                            sampler=val_sampler)

    return train_loader, val_loader


def _cifar10(root: str,
             batch_size: int = 256,
             n_views: int = 1,
             train_transform: Callable = None,
             val_transform: Callable = None) -> Tuple[DataLoader, DataLoader]:

    if not train_transform:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    if not val_transform:
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    train_transforms = _NViewTransform(train_transform, val_transform, n_views=n_views)
    val_transforms = _NViewTransform(val_transform, val_transform, n_views=1)

    train_dataset = datasets.CIFAR10(root, transform=train_transforms, train=True)
    val_dataset = datasets.CIFAR10(root, transform=val_transforms, train=False)

    return _get_loaders(train_dataset, val_dataset, batch_size)


def _cifar100(root: str,
              batch_size: int = 256,
              n_views: int = 1,
              train_transform: Callable = None,
              val_transform: Callable = None) -> Tuple[DataLoader, DataLoader]:

    if not train_transform:
        train_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

    if not val_transform:
        val_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
        ])

    train_transforms = _NViewTransform(train_transform, val_transform, n_views=n_views)
    val_transforms = _NViewTransform(val_transform, val_transform, n_views=1)

    train_dataset = datasets.CIFAR100(root, transform=train_transforms, train=True)
    val_dataset = datasets.CIFAR100(root, transform=val_transforms, train=False)

    return _get_loaders(train_dataset, val_dataset, batch_size)


def _tiny_imagenet(root: str,
                   batch_size: int = 256,
                   n_views: int = 1,
                   train_transform: Callable = None,
                   val_transform: Callable = None) -> Tuple[DataLoader, DataLoader]:

    # Different data roots for train and val
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    if not train_transform:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if not val_transform:
        val_transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_transforms = _NViewTransform(train_transform, val_transform, n_views=n_views)
    val_transforms = _NViewTransform(val_transform, val_transform, n_views=1)

    train_dataset = datasets.ImageFolder(train_root, train_transforms)
    val_dataset = datasets.ImageFolder(val_root, val_transforms)

    return _get_loaders(train_dataset, val_dataset, batch_size)


def _imagenet(root: str,
              batch_size: int = 256,
              n_views: int = 1,
              train_transform: Callable = None,
              val_transform: Callable = None) -> Tuple[DataLoader, DataLoader]:

    # Different data roots for train and val
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    if not train_transform:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    if not val_transform:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_transforms = _NViewTransform(train_transform, val_transform, n_views=n_views)
    val_transforms = _NViewTransform(val_transform, val_transform, n_views=1)

    train_dataset = datasets.ImageFolder(train_root, train_transforms)
    val_dataset = datasets.ImageFolder(val_root, val_transforms)

    return _get_loaders(train_dataset, val_dataset, batch_size)


class _NViewTransform:
    """_NViewTransform

    The transformation that creates different views of one image.
    The first view is always the validation view.

    Parameters
    ----------
    train_transform : Callable
        Train transformation.
    val_transform : Callable
        Validation transformation.
    n_views : int, optional
        Number of desired views. By default 1
    """

    def __init__(self, train_transform: Callable, val_transform: Callable, n_views: int = 1):
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.n_views = n_views

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        out = [self.val_transform(x)]
        for _ in range(self.n_views - 1):
            out.append(self.train_transform(x))
        return out


class _EmbeddingDataset(Dataset):
    """_EmbeddingDataset
    Torchvision Dataset to access embeddings. It helps to
    iterate through the embeddings and choose one random
    view whenever a specific item is indexed.

    Parameters
    ----------
    z : torch.Tensor
        3D tensor in shape (batch_size x cnn_dim x n_views).
    y : torch.Tensor
        1D tensor, the labels.
    """

    def __init__(self, z: torch.Tensor, y: torch.Tensor):
        super(_EmbeddingDataset, self).__init__()
        self.z = z
        self.y = y
        self.n_views = z.shape[-1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chosen_view = random.randint(0, self.n_views - 1)
        z = self.z[idx, :, chosen_view]
        y = self.y[idx]
        return z, y
