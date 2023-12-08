from spoter2.training import BaseTrainer
import wandb
import os
import torch


# TODO: test WandbLogger
class WandbLogger:
    def __init__(self):
        pass

    def __call__(self, trainer: BaseTrainer):
        val_loss = trainer.val_loss.compute()
        train_loss = trainer.train_loss.compute()
        if trainer.scheduler is not None:
            lr = trainer.scheduler.get_last_lr()[0]
        else:
            lr = trainer.optimizer.param_groups[0]['lr']

        wandb.log({"val_loss": val_loss, "train_loss": train_loss, "lr": lr}, step=trainer.epoch)


class SaveCheckpoint:
    def __init__(self, path: str, save_top: bool = True):
        self.last_saved_checkpoint_name = ""
        self.top_val_loss = -1
        self.path = path
        self.save_top = save_top

    def __call__(self, trainer: BaseTrainer):
        loss = trainer.val_loss.compute()
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
