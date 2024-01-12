import argparse
import os

import numpy as np
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

import wandb
from spoter2.data import StructuredDummyDataset, collate_fn, WLASLDataset
from spoter2.model import SPOTEREncoder
from spoter2.training import PretrainingTrainer, SaveCheckpoint, PretrainingPredictionExamples, WandbLogger
from spoter2.utils import set_seed, load_yaml, merge_configs


def get_args_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    # paths
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--checkpoint_folder', type=str)

    # wandb variables
    parser.add_argument('--wandb_api_key', type=str)
    parser.add_argument('--entity', type=str)
    parser.add_argument('--project', default="spoter2", type=str)
    parser.add_argument('--group', type=str)
    parser.add_argument('--tags', nargs='*', type=str)

    return parser


def spoter2_encoder_small(config):
    model = SPOTEREncoder(
        data_dim=config["data_dim"],
        hidden_dim=384,
        max_frames=config["max_sequence_length"],
        nhead=6,
        num_layers=6,
        pos_encoding=config.get("positional_encoding", "")
    )
    return model


def spoter2_encoder_base(config):
    model = SPOTEREncoder(
        data_dim=config["data_dim"],
        hidden_dim=768,
        max_frames=config["max_sequence_length"],
        nhead=12,
        num_layers=12,
        pos_encoding=config.get("positional_encoding", "")
    )
    return model


model_versions = {
    "small": spoter2_encoder_small,
    "base": spoter2_encoder_base
}


def train(config):
    set_seed(config.get("seed", 0))
    model = model_versions[config["model_name"]](config)

    if "train_file" in config and "val_file" in config:
        train_dataset = WLASLDataset(config["train_file"])
        val_dataset = WLASLDataset(config["val_file"])
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

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), config["learning_rate"])
    scheduler = None
    if config.get("scheduler", "") == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * 1e-3)
    elif config.get("scheduler", "") == "step":
        milestones = [int(np.floor(config["epochs"] * 0.5)), int(np.floor(config["epochs"] * 0.5))]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    # initialize callbacks
    callbacks = [
        PretrainingPredictionExamples(5)
    ]
    if "checkpoint_folder" in config:
        callbacks.append(SaveCheckpoint(config["checkpoint_folder"]))
    if "wandb_api_key" in config:
        os.environ['WANDB_API_KEY'] = config["wandb_api_key"]
    if "project" in config:
        # initialize wandb
        kwarg_names = ["group", "experiment", "entity", "tags"]
        wandb_kwargs = {n: config[n] for n in kwarg_names if n in config}

        wandb.init(
            project=config["project"],
            config=config,
            **wandb_kwargs
        )
        callbacks.append(WandbLogger())

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

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('', parents=[get_args_parser()])
    args = parser.parse_args()

    config = load_yaml(args.config_file)
    config = merge_configs(config, vars(args))

    train(config)
