from spoter2.training import BaseTrainer
from tqdm import tqdm
from torchmetrics import MeanMetric
import torch


class PretrainingTrainer(BaseTrainer):
    def __init__(self, mask_ratio: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.mask_ratio = mask_ratio

    def __call__(self, data, padding_idx, mask_ratio):
        if data.device != self.device:
            data = data.to(self.device)
        prediction = self.model(data, padding_idx, mask_ratio)
        return prediction

    def loss_calculation(self, predictions, targets):
        batch_loss = []
        for i, (prd, trg) in enumerate(zip(predictions, targets)):
            batch_loss.append(self.criterion(prd, trg))
        return torch.mean(torch.stack(batch_loss))

    def train(self):
        train_loss = []
        val_loss = []
        for epoch in range(self.epochs):
            self.epoch = epoch
            train_epoch_loss = self.train_epoch(self.train_loader)
            val_epoch_loss = self.validate_epoch(self.val_loader)
            train_loss.append(train_epoch_loss)
            val_loss.append(val_epoch_loss)

            self.apply_callbacks()

        return train_loss, val_loss

    def train_epoch(self, dataloader):
        self.model.train()
        self.train_loss.reset()

        pbar = tqdm(dataloader, desc=f"{self.epoch + 1}/{self.epochs}")
        for _, data in enumerate(pbar):
            data["data"] = data["data"].to(self.device)

            # forward pass
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                predictions, targets = self(data["data"], data["padding_idx"], self.mask_ratio)
                batch_loss = self.loss_calculation(predictions, targets)

            # backward pass
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            # update metrics
            self.train_loss.update(batch_loss)
            pbar.set_description(f"{self.epoch + 1}/{self.epochs}: Train Loss: {self.train_loss.compute().item():.4f}")
        return self.train_loss.compute().item()

    def validate_epoch(self, dataloader):
        self.model.eval()
        self.val_loss.reset()

        pbar = tqdm(dataloader, desc=f"{self.epoch + 1}/{self.epochs}")
        for _, data in enumerate(pbar):
            data["data"] = data["data"].to(self.device)

            # forward pass
            with torch.cuda.amp.autocast():
                predictions, targets = self(data["data"], data["padding_idx"], self.mask_ratio)
                batch_loss = self.loss_calculation(predictions, targets)

            # update metrics
            self.val_loss.update(batch_loss)
            pbar.set_description(f"{self.epoch + 1}/{self.epochs}: Val Loss: {self.val_loss.compute().item():.4f}")
        return self.val_loss.compute().item()
