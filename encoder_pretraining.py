import argparse
from datetime import datetime
import os

from torch import nn

from spoter2.data import StructuredDummyDataset, WLASLDataset, How2SignDataset
from spoter2.model import SPOTEREncoderPreTraining
from spoter2.training import (PretrainingPredictionExamples,
                              PretrainingTrainer, SaveCheckpoint)
from spoter2.utils import load_yaml, merge_configs, set_seed
from training_utils import get_scheduler, get_optimizer, get_dataloaders, get_wandb_callback, get_default_parser


def get_args_parser():
    parser = get_default_parser()

    return parser


def spoter2_encoder_small(config):
    model = SPOTEREncoderPreTraining(
        data_dim=config["data_dim"],
        hidden_dim=384,
        max_frames=config["max_sequence_length"],
        nhead=6,
        num_layers=6,
        pos_encoding=config.get("positional_encoding", "")
    )
    return model


def spoter2_encoder_base(config):
    model = SPOTEREncoderPreTraining(
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

    dataset_name = config.get("dataset_name", "").lower()
    if dataset_name == "wlasl":
        train_dataset = WLASLDataset(config["train_file"])
        val_dataset = WLASLDataset(config["val_file"])
    elif dataset_name == "how2sign":
        train_dataset = How2SignDataset(config["train_file"], config.get("train_video_path", None),
                                        kp_normalization=config.get("kp_normalization", []))
        val_dataset = How2SignDataset(config["val_file"], config.get("train_video_path", None),
                                      kp_normalization=config.get("kp_normalization", []))
    else:
        train_dataset = StructuredDummyDataset(256, (128, 256), config["data_dim"])
        val_dataset = StructuredDummyDataset(256, (128, 256), config["data_dim"])

    train_loader, val_loader = get_dataloaders(config, train_dataset, val_dataset)

    criterion = nn.MSELoss()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # initialize callbacks
    callbacks = [
        PretrainingPredictionExamples(5)
    ]
    if config.get("checkpoint_folder", ""):
        experiment_name = datetime.now().strftime("%d-%m_%H-%M-%S")
        if config.get("name", ""):
            experiment_name = f"{config['name']}-{experiment_name}"
        config["checkpoint_folder"] = os.path.join(config["checkpoint_folder"], experiment_name)

        callbacks.append(SaveCheckpoint(config["checkpoint_folder"]))

    wandb_callback = get_wandb_callback(config)
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

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
