import argparse
from typing import Any, Dict, Union
import warnings
import yaml

from probnmn.config import Config


def add_common_args(parser):
    parser.add_argument(
        "--config-override",
        default=[],
        nargs="*",
        help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
             "nesting) using a dot operator. The actual config will be updated and recorded in "
             "the serialization directory."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed for all devices, useful for doing multiple runs and reporting mean and "
             "variance."
    )

    parser.add_argument_group("CLEVR Data files.")
    parser.add_argument(
        "--tokens-train-h5",
        default="data/clevr_tokens_train.h5",
        help="Path to HDF file containing tokenized CLEVR v1.0 train split programs, questions "
             "and answers, and corresponding image indices."
    )
    parser.add_argument(
        "--tokens-val-h5",
        default="data/clevr_tokens_val.h5",
        help="Path to HDF file containing tokenized CLEVR v1.0 val split programs,"
             " questions and answers, and corresponding image indices."
    )
    parser.add_argument(
        "--features-train-h5",
        default="data/features_train.h5",
        help="Path to HDF file containing pre-extracted features from CLEVR v1.0 train images."
    )
    parser.add_argument(
        "--features-val-h5",
        default="data/features_val.h5",
        help="Path to HDF file containing pre-extracted features from CLEVR v1.0 val images."
    )
    parser.add_argument(
        "--vocab-dirpath",
        default="data/clevr_vocab",
        help="Path to directory containing vocabulary for programs, questions and"
             " answers."
    )

    parser.add_argument_group("Compute resource controlling arguments.")
    parser.add_argument(
        "--gpu-ids",
        default=[0],
        nargs="+",
        type=int,
        help="List of ids of GPUs to use (-1 for CPU)."
    )
    parser.add_argument(
        "--num-val-examples",
        default=10000,
        type=int,
        help="Number of validation examples to use. CLEVR val is huge, this can be used to make "
             "the validation loop a bit faster, although might provide a noisy estimate of "
             "performance."
    )

    parser.add_argument_group("Checkpointing related arguments")
    parser.add_argument(
        "--save-dirpath",
        default="checkpoints/experiment",
        help="Path of directory to save checkpoints, this path is recommended to be empty or "
             "non-existent. Having previously saved checkpoints in this directory might overwrite "
             "them."
    )
    parser.add_argument(
        "--checkpoint-every",
        default=500,
        type=int,
        help="Save a checkpoint after every this many epochs/iterations."
    )


def read_config(config_ymlpath: str):
    config = yaml.load(open(config_ymlpath))
    return config


def override_config_from_opts(config: Union[Config, Dict[str, Any]],
                              config_override_str: str):
    # Convert string to a python dict.
    config_override: Dict[str, Any] = eval(config_override_str)

    for config_key in config_override:
        if config_key in config:
            config[config_key] = config_override[config_key]
        else:
            warnings.warn(f"Config {config_key}, does not exist in provided config file.")
    return config


def print_config_and_args(config: Union[Config, Dict[str, Any]],
                          args: argparse.Namespace):
    print(config)
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))


def cycle(iterable):
    # Using itertools.cycle with dataloader is harmful
    while True:
        for x in iterable:
            yield x
