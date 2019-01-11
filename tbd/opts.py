from typing import Optional


def add_common_opts(parser):
    parser.add_argument_group("CLEVR Data files.")
    parser.add_argument(
        "--tokens-train-h5",
        default="data/clevr_tokens_train.h5",
        help="Path to HDF file containing tokenized CLEVR v1.0 train split programs,"
             " questions and answers, and corresponding image indices.",
    )
    parser.add_argument(
        "--tokens-val-h5",
        default="data/clevr_tokens_val.h5",
        help="Path to HDF file containing tokenized CLEVR v1.0 val split programs,"
             " questions and answers, and corresponding image indices.",
    )
    parser.add_argument(
        "--vocab-dirpath",
        default="data/clevr_vocab",
        help="Path to directory containing vocabulary for programs, questions and"
             " answers.",
    )

    parser.add_argument_group("Compute resource controlling arguments.")
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        help="List of ids of GPUs to use (-1 for CPU).",
    )

    parser.add_argument_group("Checkpointing related arguments")
    parser.add_argument(
        "--save-dirpath",
        default="checkpoints/experiment",
        help="Path of directory to save checkpoints, this path is recommended to be empty"
             " or non-existent. Having previously saved checkpoints in this directory might"
             " overwrite them.",
    )
    parser.add_argument(
        "--checkpoint-every",
        default=500,
        type=int,
        help="Save a checkpoint after every this many epochs/iterations.",
    )
