import argparse

from torch import nn

from spoter2.data import StructuredDummyDataset, WLASLDataset
from spoter2.model import SPOTERClassification
from spoter2.training import ClassificationTrainer, SaveCheckpoint
from spoter2.utils import load_yaml, merge_configs, set_seed
from training_utils import get_scheduler, get_optimizer, get_dataloaders, get_wandb_callback, get_default_parser


def get_args_parser():
    parser = get_default_parser()

    return parser


def spoter2_classification_small(config):
    model = SPOTERClassification(
        data_dim=config["data_dim"],
        hidden_dim=384,
        max_frames=config["max_sequence_length"],
        nhead=6,
        num_layers=6,
        pos_encoding=config.get("positional_encoding", ""),
        num_classes=100
    )
    return model


def spoter2_classification_base(config):
    model = SPOTERClassification(
        data_dim=config["data_dim"],
        hidden_dim=768,
        max_frames=config["max_sequence_length"],
        nhead=12,
        num_layers=12,
        pos_encoding=config.get("positional_encoding", ""),
        num_classes=100
    )
    return model


model_versions = {
    "small": spoter2_classification_small,
    "base": spoter2_classification_base
}


def train(config):
    set_seed(config.get("seed", 0))
    model = model_versions[config["model_name"]](config)
    if config.get("checkpoint", ""):
        model.load_encoder(config["checkpoint"])

    if "train_file" in config and "val_file" in config:
        train_dataset = WLASLDataset(config["train_file"])
        val_dataset = WLASLDataset(config["val_file"])
    else:
        train_dataset = StructuredDummyDataset(256, (128, 256), config["data_dim"])
        val_dataset = StructuredDummyDataset(256, (128, 256), config["data_dim"])

    train_loader, val_loader = get_dataloaders(config, train_dataset, val_dataset)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    # initialize callbacks
    callbacks = []
    if config.get("checkpoint_folder", ""):
        callbacks.append(SaveCheckpoint(config["checkpoint_folder"]))

    wandb_callback = get_wandb_callback(config)
    if wandb_callback is not None:
        callbacks.append(wandb_callback)

    trainer = ClassificationTrainer(
        num_classes=100,
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
