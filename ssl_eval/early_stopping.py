from copy import deepcopy


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, delta=0, restore_best=True):
        self.patience = patience
        self.delta = delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_loss = None
        self.best_model = None
        self.best_model

    def __call__(self, val_loss, model):

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
