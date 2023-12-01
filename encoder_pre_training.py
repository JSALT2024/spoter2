from spoter2.model import SPOTEREncoder
from spoter2.data import StructuredDummyDataset, collate_fn
from spoter2.training import PretrainingTrainer, BaseTrainer
import torch
from torch import nn, optim

from torch.utils.data import DataLoader
from functools import partial


def main(config):
    model = SPOTEREncoder(108, 256, 9, 6, config["positional_encoding"])

    train_dataset = StructuredDummyDataset(64, (128, 256), 108)
    val_dataset = StructuredDummyDataset(8, (128, 256), 108)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=partial(
            collate_fn,
            pad_token=torch.zeros([1, 108]),
            mask_prob=config["mask_probability"]
        )
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=partial(
            collate_fn,
            pad_token=model.pad_token,
            mask_prob=config["mask_probability"]
        )
    )

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), config["learning_rate"])

    trainer = PretrainingTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer
    )


if __name__ == "__main__":
    basic_config = {
        "learning_rate": 0.001,
        "batch_size": 8,
        "epochs": 12,
        "mask_probability": 0.2,
        "positional_encoding": "learnable_uniform"

    }
    main(basic_config)
