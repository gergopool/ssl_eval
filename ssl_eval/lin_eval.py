import torch
import pkbar
from torch import nn
from .distributed import AllGather, AllReduce

from .larc import LARC
from .early_stopping import EarlyStopping
from .data import create_lin_eval_dataloader
from .utils import DistributedAverageMeter


class LinClassifier(nn.Module):

    def __init__(self, cnn_dim, n_classes):
        super(LinClassifier, self).__init__()
        self.bn = nn.BatchNorm1d(cnn_dim)
        self.fc = nn.Linear(cnn_dim, n_classes)
        self.init_weights()

    def init_weights(self):
        self.fc.weight.data.normal_(mean=0.0, std=0.01)
        self.fc.bias.data.zero_()

    def forward(self, x):
        return self.fc(self.bn(x))


class LinearEvaluator:

    def __init__(self, cnn_dim, n_classes, device, warm_start=True):
        self.cnn_dim = cnn_dim
        self.n_classes = n_classes
        self.device = device
        self.warm_start = warm_start

        self.classifier = None
        self._reset_classifier()

    def _reset_classifier(self):
        model = LinClassifier(self.cnn_dim, self.n_classes).to(self.device)
        if self.world_size > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[self.device], output_device=self.device)
        self.classifier = model

    def train(self,
              train_z: torch.Tensor,
              train_y: torch.Tensor,
              epochs: int = 100,
              batch_size: int = 256,
              lr: float = 0.1):

        if not self.warm_start:
            self._reset_classifier()

        self.classifier.train()

        opt = torch.optim.SGD(self.classifier.parameters(), lr=lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                               patience=5,
                                                               min_lr=lr / 100.)
        opt = LARC(opt, trust_coefficient=0.001, clip=False)

        early_stopper = EarlyStopping(patience=15, restore_best=True)

        data_loader = create_lin_eval_dataloader(train_z, train_y, batch_size)

        if self.verbose:
            bs = batch_size * self.world_size
            print(f"\nLinear Eval - params: lr={lr:0.5f} | batch_size={bs} | " +
                  f"num_GPUs: {self.world_size}")
            print(f"Linear Eval - Training {epochs} epochs.")
            pbar = pkbar.Kbar(target=epochs)  # Progress bar

        for epoch in range(epochs):

            if self.world_size > 1:
                train_loader.sampler.set_epoch(epoch)

    def run_epoch(self, data_laoder, criterion, opt=None):
        loss_meter = DistributedAverageMeter(self.device)
        acc_meter = DistributedAverageMeter(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)

        for z, y in data_laoder:
            z = z.to(self.device)
            y = y.to(self.device)
            y_hat = self.classifier(z, y)
            loss = criterion(y_hat, y)

            n_hits = (y_hat.argmax(dim=1) == y).sum()
            loss_meter.update(loss.item(), n=1)
            acc_meter.update(n_hits, len(y))
