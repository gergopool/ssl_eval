from torch import nn
from copy import deepcopy


class EarlyStopping:
    """EarlyStopping
    Early stops the training if validation loss doesn't improve after a given patience.

    Parameters
    ----------
    patience : int, optional
        Number of epochs to wait without improvement, by default 15
    delta : int, optional
        A difference to the best value which makes the new best value an improvement, by default 0
    """

    def __init__(self, patience: int = 15, delta: int = 0):
        self.patience = patience
        self.delta = delta

        # Counting how many times we could not improve by a delta differnce
        self.counter = 0

        # Best values
        self.best_loss = None
        self.best_model = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:

        # First run
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = deepcopy(model)
            return False

        # Out of delta distance
        if val_loss < self.best_loss - self.delta:
            self.counter = 0
        else:
            self.counter += 1

        # Still, if this is the best model, save
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = deepcopy(model)

        return self.counter >= self.patience
