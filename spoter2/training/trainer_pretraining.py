from spoter2.training import BaseTrainer
from tqdm import tqdm
from torchmetrics import MeanMetric
import torch


class PretrainingTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.train_epoch(self.train_loader)

    def __call__(self, batch_data):
        if batch_data.device != self.device:
            batch_data = batch_data.to(self.device)
        prediction = self.model(batch_data)
        return prediction

    def loss_calculation(self, prediction, data):
        batch_size = data["data"].shape[1]
        batch_loss = []
        for b in range(batch_size):
            batch_mask_idxs = data["mask_idxs"][b]
            batch_targets = data["target"][b]
            batch_preds = prediction[batch_mask_idxs, b, :]

            batch_loss.append(self.criterion(batch_preds, batch_targets))
        return torch.mean(torch.tensor(batch_loss))

    def train_epoch(self, dataloader):
        self.model.train()
        loss = MeanMetric().to(self.device)

        pbar = tqdm(dataloader, desc=f"{self.epoch + 1}/{self.epochs}")
        for _, data in enumerate(pbar):
            data["data"] = data["data"].to(self.device)
            batch_data = data["data"]

            self.optimizer.zero_grad(set_to_none=True)

            # forward pass
            with torch.cuda.amp.autocast():
                prediction = self(batch_data)

            batch_loss = self.loss_calculation(prediction, data)
            # # update metrics
            loss.update(batch_loss)

            # # backward pass + weight update
            self.scaler.scale(batch_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def validate_epoch(self, dataloader):
        self.model.eval()
        pass
