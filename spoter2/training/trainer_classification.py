from spoter2.training import BaseTrainer
from tqdm import tqdm
import torch
import wandb
from torchmetrics import Accuracy


class ClassificationTrainer(BaseTrainer):
    def __init__(self, num_classes: int, **kwargs):
        super().__init__(**kwargs)
        self.metrics["val_accuracy"] = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.metrics["train_accuracy"] = Accuracy(task="multiclass", num_classes=num_classes).to(self.device)

    def __call__(self, data, padding_idx):
        if data.device != self.device:
            data = data.to(self.device)
        prediction = self.model(data, padding_idx)
        return prediction

    def loss_calculation(self, predictions, targets):
        return self.criterion(predictions, targets)

    def train_epoch(self, dataloader):
        self.model.train()

        pbar = tqdm(dataloader, desc=f"{self.epoch + 1}/{self.epochs}")
        for _, data in enumerate(pbar):
            data["data"] = data["data"].to(self.device)
            data["label"] = data["label"].to(self.device)

            # forward pass
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                predictions = self(data["data"], data["padding_idx"])
                prediction_labels = torch.argmax(predictions, 1)
                batch_loss = self.loss_calculation(predictions, data["label"])

            # backward pass
            self.backward_pass(batch_loss)

            # update metrics
            self.metrics["train_loss"].update(batch_loss)
            self.metrics["train_accuracy"].update(prediction_labels, data["label"])
            pbar.set_description(
                f"{self.epoch + 1}/{self.epochs}: Train Loss: {self.metrics['train_loss'].compute().item():.4f} "
                f"Train Accuracy: {self.metrics['train_accuracy'].compute().item():.4f}")
        return self.metrics["train_loss"].compute().item()

    def validate_epoch(self, dataloader):
        self.model.eval()

        pbar = tqdm(dataloader, desc=f"{self.epoch + 1}/{self.epochs}")
        for _, data in enumerate(pbar):
            data["data"] = data["data"].to(self.device)
            data["label"] = data["label"].to(self.device)

            # forward pass
            with torch.cuda.amp.autocast():
                predictions = self(data["data"], data["padding_idx"])
                prediction_labels = torch.argmax(predictions, 1)
                batch_loss = self.loss_calculation(predictions, data["label"])

            # update metrics
            self.metrics["val_loss"].update(batch_loss)
            self.metrics["val_accuracy"].update(prediction_labels, data["label"])
            pbar.set_description(
                f"{self.epoch + 1}/{self.epochs}: Val Loss: {self.metrics['val_loss'].compute().item():.4f} "
                f"Val Accuracy: {self.metrics['val_accuracy'].compute().item():.4f}")
        return self.metrics["val_loss"].compute().item()
