import torch
from torchmetrics import MeanMetric
from typing import Protocol
from spoter2.base_logger import logger
import wandb


class Callback(Protocol):
    def __init__(self):
        ...

    def before_training(self):
        ...

    def __call__(self, trainer):
        ...


class BaseTrainer:
    def __init__(
            self,
            epochs: int,
            model: torch.nn.Module,
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

        self.metrics = {
            "val_loss": MeanMetric().to(self.device),
            "train_loss": MeanMetric().to(self.device)
        }

    def before_training_callbacks(self):
        if self.callbacks is None:
            return

        logger.info("Callback info:")
        for callback in self.callbacks:
            callback.before_training()

    def apply_callbacks(self):
        if self.callbacks is None:
            return

        for callback in self.callbacks:
            callback(self)

    def backward_pass(self, batch_loss):
        self.scaler.scale(batch_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()

    def train_epoch(self, dataloader) -> float:
        raise NotImplemented()

    def validate_epoch(self, dataloader) -> float:
        raise NotImplemented()

    def train(self):
        self.before_training_callbacks()

        train_loss = []
        val_loss = []
        for epoch in range(self.epochs):
            self.epoch = epoch
            [m.reset() for m in self.metrics.values()]

            train_epoch_loss = self.train_epoch(self.train_loader)
            val_epoch_loss = self.validate_epoch(self.val_loader)
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)

            self.apply_callbacks()

        if wandb.run is not None:
            wandb.finish()

        return train_loss, val_loss

