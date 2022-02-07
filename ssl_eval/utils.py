import torch
from .distributed import AllReduce


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """_accuracy 
    Accuracy of the model
    """
    pred = torch.argmax(y_hat, dim=1)
    acc = (pred == y).sum() / len(y)
    return AllReduce.apply(acc)