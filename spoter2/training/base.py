import torch
from torchmetrics import MeanMetric
from typing import Protocol


class Callback(Protocol):
    def __init__(self):
        ...

    def __call__(self, trainer):
        ...


class BaseTrainer:
    def __init__(
            self,
            epochs: int,
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler=None,
            callbacks: None | list[Callback] = None
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.epoch = 0
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = torch.cuda.amp.GradScaler()
        self.callbacks = callbacks

        self.val_loss = MeanMetric().to(self.device)
        self.train_loss = MeanMetric().to(self.device)

    def apply_callbacks(self):
        if self.callbacks is None:
            return

        for callback in self.callbacks:
            callback(self)
