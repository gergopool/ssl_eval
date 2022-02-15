import os
import random
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms, datasets
from typing import Tuple

from .distributed import get_world_size_n_rank

__all__ = ['get_loaders_by_name', "create_lin_eval_dataloader"]


def get_loaders_by_name(root: str, dataset_name: str, **kwargs):
    if dataset_name in ['imagenet', 'cifar10', 'cifar100']:
        return globals()[dataset_name](root, **kwargs)
    else:
        raise NameError(
            f"Unknown dataset name: {dataset_name}. " +\
             "Please choose from [imagenet, cifar10, cifar100]"
        )


def get_loaders(train_dataset: Dataset,
                val_dataset: Dataset,
                batch_size: int = 256) -> Tuple[DataLoader, DataLoader]:

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


def cifar10(root: str, batch_size: int = 256, n_views: int = 1):

    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_trans = NViewTransform(trans, n=n_views)

    train_dataset = datasets.CIFAR10(root, transform=train_trans, train=True)
    val_dataset = datasets.CIFAR10(root, transform=trans, train=False)

    return get_loaders(train_dataset, val_dataset, batch_size)


def cifar100(root: str, batch_size: int = 256, n_views: int = 1):

    trans = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])
    ])

    train_trans = NViewTransform(trans, n=n_views)

    train_dataset = datasets.CIFAR100(root, transform=train_trans, train=True)
    val_dataset = datasets.CIFAR100(root, transform=trans, train=False)

    return get_loaders(train_dataset, val_dataset, batch_size)


def imagenet(root: str, batch_size: int = 256, n_views: int = 1):

    # Different data roots for train and val
    train_root = os.path.join(root, 'train')
    val_root = os.path.join(root, 'val')

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_trans = NViewTransform(train_trans, n=n_views)

    train_dataset = datasets.ImageFolder(train_root, train_trans)
    val_dataset = datasets.ImageFolder(val_root, val_trans)

    return get_loaders(train_dataset, val_dataset, batch_size)


class NViewTransform:

    def __init__(self, transform, n=1):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        out = []
        for _ in range(self.n):
            out.append(self.transform(x))
        return out


# DATA FOR LINEAR EVAL


class EmbeddingBank(Dataset):

    def __init__(self, z, y):
        super(EmbeddingBank, self).__init__()
        self.z = z
        self.y = y
        self.n_views = z.shape[-1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        chosen_view = random.randint(0, self.n_views - 1)
        z = self.z[idx, :, chosen_view]
        y = self.y[idx]

        return z, y


def create_lin_eval_dataloader(z, y, batch_size=1000):

    emb_dataset = EmbeddingBank(z, y)

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