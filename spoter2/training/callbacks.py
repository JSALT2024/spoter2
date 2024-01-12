from spoter2.training import BaseTrainer
import wandb
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from spoter2.base_logger import logger


class WandbLogger:
    def __init__(self):
        pass

    def before_training(self):
        logger.info(f"{self.__class__.__name__}: Logging using wandb.")

    def __call__(self, trainer: BaseTrainer):
        metrics = {name: metric.compute() for name, metric in trainer.metrics.items()}
        if trainer.scheduler is not None:
            lr = trainer.scheduler.get_last_lr()[0]
        else:
            lr = trainer.optimizer.param_groups[0]['lr']

        metrics["lr"] = lr
        wandb.log(metrics, step=trainer.epoch)


class SaveCheckpoint:
    def __init__(self, path: str, save_top: bool = True):
        self.last_saved_checkpoint_name = ""
        self.top_val_loss = -1
        self.path = path
        self.save_top = save_top

    def before_training(self):
        logger.info(f"{self.__class__.__name__}: Save directory: {self.path}.")

    def __call__(self, trainer: BaseTrainer):
        loss = trainer.metrics["val_loss"].compute()
        if self.top_val_loss < 0 or loss <= self.top_val_loss:
            self.top_val_loss = loss

            if not os.path.isdir(self.path):
                os.makedirs(self.path, exist_ok=True)
            path = os.path.join(self.path, f"checkpoint-ep_{trainer.epoch}-val_loss_{loss:.05f}.pth")

            components_to_save = {
                'epoch': trainer.epoch,
                'model': trainer.model.state_dict(),
                'optimizer': trainer.optimizer.state_dict(),
            }
            if trainer.scheduler is not None:
                components_to_save['scheduler'] = trainer.scheduler.state_dict()
            torch.save(components_to_save, path)

            if os.path.isfile(self.last_saved_checkpoint_name):
                os.remove(self.last_saved_checkpoint_name)
            self.last_saved_checkpoint_name = path


class PretrainingPredictionExamples:
    def __init__(self, period=1):
        self.period = period
        self.last_epoch = 0
        self.fig_size = 2.5

    def before_training(self):
        if wandb.run is None:
            logger.warning(f"{self.__class__.__name__}: Wandb was not initialized. Example images will not be saved.")

    def __call__(self, trainer: BaseTrainer):

        if trainer.epoch - self.last_epoch == self.period - 1:
            # get data
            batch = next(iter(trainer.train_loader))
            batch["data"] = batch["data"].to(trainer.device)

            # predict
            trainer.model.eval()
            predictions, targets, mask_idxs = trainer.model(
                batch["data"].clone(),
                batch["padding_idx"],
                trainer.mask_ratio,
                get_mask_idx=True
            )
            targets = [t.detach().cpu() for t in targets]
            predictions = [p.detach().cpu() for p in predictions]

            fig, ax = plt.subplots(
                len(targets),
                5,
                figsize=(5 * self.fig_size, len(targets) * self.fig_size),
                width_ratios=[3, 3, 1, 1, 1]
            )

            # plot full prediction
            for i, (_t, _p) in enumerate(zip(targets, predictions)):
                ax[i, 0].imshow(_t)
                ax[i, 1].imshow(_p)

            # plot individual frames
            for i, (data, padding_idx, prediction, mask_idx) in enumerate(
                    zip(batch["data"], batch["padding_idx"], predictions, mask_idxs)):
                idx_select = [0, int(np.round(len(prediction) / 2)), len(prediction) - 1]

                data = data[:padding_idx].detach().cpu().numpy()[mask_idx]
                prediction = prediction.detach().cpu().numpy()

                selected_frames_data = data[idx_select]
                selected_frames_prediction = prediction[idx_select]
                for c, (d, p) in enumerate(zip(selected_frames_data, selected_frames_prediction)):
                    dx, dy = d[0::2], d[1::2]
                    px, py = p[0::2], p[1::2]
                    ax[i, c + 2].scatter(dx, dy, c="tab:green", marker="x", label="target")
                    ax[i, c + 2].scatter(px, py, c="tab:red", marker="+", label="prediction")
                    ax[i, c + 2].set(xlim=(0, 1), ylim=(0, 1))
                    ax[i, c + 2].set_aspect("equal")
                    ax[i, c + 2].set_title(f"frame: {idx_select[c]}")
                    ax[i, c + 2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.2))

            ax[0, 0].set_title("target")
            ax[0, 1].set_title("prediction")
            fig.tight_layout()
            plt.close("all")

            if wandb.run is not None:
                wandb.log({"example predictions": fig})

            self.last_epoch = trainer.epoch
