import argparse
import os

import numpy as np
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader

import wandb
from spoter2.data import collate_fn
from spoter2.training import WandbLogger


def get_default_parser():
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
    parser.add_argument('--name', type=str)
    parser.add_argument('--tags', nargs='*', type=str)

    return parser


def get_optimizer(config, model):
    if config.get("optimizer", "").lower() == "adamw":
        optimizer = optim.AdamW(model.parameters(), config["learning_rate"])
    else:
        raise ValueError(f'Unknown optimizer: {config.get("optimizer", "")}')

    return optimizer


def get_scheduler(config, optimizer):
    scheduler = None
    if config.get("scheduler", "") == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] * 1e-3)
    elif config.get("scheduler", "") == "step":
        milestones = [int(np.floor(config["epochs"] * 0.5)), int(np.floor(config["epochs"] * 0.5))]
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    return scheduler


def get_dataloaders(config, train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=config["batch_size"],
        collate_fn=collate_fn,
        num_workers=config["num_workers"]
    )

    return train_loader, val_loader


def get_wandb_callback(config):
    wandb_callback = None
    if "wandb_api_key" in config:
        os.environ['WANDB_API_KEY'] = config["wandb_api_key"]

    if config.get("project", "") and os.environ['WANDB_API_KEY']:
        # initialize wandb
        kwarg_names = ["group", "name", "entity", "tags"]
        wandb_kwargs = {n: config[n] for n in kwarg_names if n in config}

        wandb.init(
            project=config["project"],
            config=config,
            **wandb_kwargs
        )
        wandb_callback = WandbLogger()

    return wandb_callback
