import argparse
import os
from functools import partial

import wandb
from encoder_pretraining import get_args_parser, train
from spoter2.utils import load_yaml, merge_configs


def get_sweep_args_parser():
    parser = get_args_parser()
    # sweep variables
    parser.add_argument('--sweep_config_file', type=str)
    parser.add_argument('--sweep_id', type=str)

    return parser


def get_args(get_parser):
    parser = argparse.ArgumentParser('', parents=[get_parser()])
    args = parser.parse_args()
    return args


def main(config=None):
    with wandb.init():
        sweep_config = wandb.config
        for variable_name in sweep_config.keys():
            config[variable_name] = sweep_config[variable_name]

        train(config)


if __name__ == "__main__":
    args = get_args(get_sweep_args_parser)

    # get config
    config = load_yaml(args.config_file)
    config = merge_configs(config, vars(args))
    if config.get("tags", None) is None:
        config["tags"] = ["sweep"]
    else:
        config["tags"].append("sweep")
        config["tags"] = list(set(config["tags"]))

    # start sweep
    if "wandb_api_key" in config:
        os.environ['WANDB_API_KEY'] = config["wandb_api_key"]

    sweep_config = load_yaml(args.sweep_config_file)
    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        # initialize wandb
        kwarg_names = ["group", "experiment", "entity", "tags"]
        wandb_kwargs = {n: config[n] for n in kwarg_names if n in config}
        sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, **wandb_kwargs)
    wandb.agent(sweep_id, partial(main, config), count=100)
    wandb.finish()
