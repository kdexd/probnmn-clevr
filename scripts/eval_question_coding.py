import argparse
from typing import Any, Dict, Optional

from allennlp.data import Vocabulary
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.data import QuestionCodingDataset
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
import probnmn.utils.checkpointing as checkpointing_utils
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Question coding for CLEVR v1.0 programs and questions.")
parser.add_argument(
    "--config-yml",
    default="configs/question_coding.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--split", choices=["train", "val"],
    default="train",
    help="Which split to evaluate on, (default: train) because that's the most important"
         "for better module training."
)
parser.add_argument(
    "--pg-checkpoint-pthpath",
    default="data/program_generator_best.pth",
    help="Path to program generator pre-trained checkpoint."
)
parser.add_argument(
    "--qr-checkpoint-pthpath",
    default="data/question_reconstructor_best.pth",
    help="Path to question reconstructor pre-trained checkpoint."
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)
args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(args.random_seed)


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    config = common_utils.read_config(args.config_yml)
    config = common_utils.override_config_from_opts(config, args.config_override)
    common_utils.print_config_and_args(config, args)

    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL
    # ============================================================================================

    vocabulary = Vocabulary.from_files(args.vocab_dirpath)
    if args.split == "train":
        eval_dataset = QuestionCodingDataset(args.tokens_train_h5)
    else:
        eval_dataset = QuestionCodingDataset(args.tokens_val_h5)

    batch_size = config["optim_batch_size"]
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size)

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
    ).to(device)
    program_generator_model, _ = checkpointing_utils.load_checkpoint(args.pg_checkpoint_pthpath)
    program_generator.load_state_dict(program_generator_model)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
    ).to(device)
    question_reconstructor_model, _ = checkpointing_utils.load_checkpoint(args.qr_checkpoint_pthpath)
    question_reconstructor.load_state_dict(question_reconstructor_model)

    program_generator.eval()
    question_reconstructor.eval()

    for i, batch in enumerate(tqdm(eval_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            # We really only need to accumulate metrics in these classes.
            _ = program_generator(batch["question"], batch["program"])
            _ = question_reconstructor(batch["program"], batch["question"])

    __pg_metrics = program_generator.get_metrics()
    __qr_metrics = question_reconstructor.get_metrics()

    # keys: {"BLEU", "perplexity", "sequence_accuracy"}
    for metric_name in __pg_metrics:
        print(f"Program generator, {metric_name}:", __pg_metrics[metric_name])
        print(f"Question reconstructor, {metric_name}:", __qr_metrics[metric_name])
        print("\n")
