import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

from spoter2.data import StructuredDummyDataset, collate_fn, WLASLDataset
from spoter2.model import SPOTEREncoder
from spoter2.training import PretrainingTrainer, SaveCheckpoint
from spoter2.utils import set_seed, plot_batch


def main(config):
    set_seed(config.get("seed", 0))
    model = SPOTEREncoder(config["data_dim"], config["feature_dim"], 256, 8, 6, config.get("positional_encoding", ""))

    if "train_path" in config and "val_path" in config:
        train_dataset = WLASLDataset(config["train_path"])
        val_dataset = WLASLDataset(config["val_path"])
    else:
        train_dataset = StructuredDummyDataset(256, (128, 256), config["data_dim"])
        val_dataset = StructuredDummyDataset(256, (128, 256), config["data_dim"])

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=collate_fn
    )

    batch = next(iter(val_loader))
    data = batch["data"]
    plot_batch(data)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), config["learning_rate"])
    scheduler = None
    if config.get("scheduler", "") == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * 1e-3)
    elif config.get("scheduler", "") == "step":
        milestones = [int(np.floor(config["epochs"] * 0.5)), int(np.floor(config["epochs"] * 0.5))]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    callbacks = []
    if "checkpoint_folder" in config:
        callbacks.append(SaveCheckpoint(config["checkpoint_folder"]))

    trainer = PretrainingTrainer(
        mask_ratio=config.get("mask_ratio", 0.1),
        epochs=config["epochs"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks
    )

    train_loss, val_loss = trainer.train()

    # plot loss
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.grid()
    plt.show()

    # plot results
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = next(iter(val_loader))
    data["data"] = data["data"].to(device)
    model.eval()
    predictions, targets = model(data["data"], data["padding_idx"], config["mask_ratio"])
    fig, ax = plt.subplots(2, len(targets), figsize=(len(targets) * 5, 2 * 2))
    for i, (_t, _p) in enumerate(zip(targets, predictions)):
        ax[0, i].imshow(_t.detach().cpu())
        ax[1, i].imshow(_p.detach().cpu())
    ax[0, 0].set_ylabel("target")
    ax[1, 0].set_ylabel("prediction")
    plt.show()


if __name__ == "__main__":
    basic_config = {
        "learning_rate": 0.0001,
        "batch_size": 8,
        "epochs": 10,
        "mask_ratio": 0.5,
        "positional_encoding": "learnable_normal",
        "seed": 0,
        "scheduler": "cos",
        "feature_dim": 256,
        "data_dim": 110,
        "train_path": "D:\\Datastes\\WLASL\\WLASL100_train_25fps.csv",
        "val_path": "D:\\Datastes\\WLASL\\WLASL100_val_25fps.csv",
        "checkpoint_folder": r"D:\MLProjects\JSALT\spoter\spoter2_checkpoints\dbg_call"
    }
    main(basic_config)
