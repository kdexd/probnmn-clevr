import argparse
import os

from allennlp.data import Vocabulary
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data import QuestionCodingDataset
from probnmn.models import ProgramGenerator, QuestionReconstructor
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
    "--qc-checkpoint-pthpath",
    default="data/question_coding_best.pth",
    help="Path to question coding pre-trained checkpoint."
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)
    common_utils.print_config_and_args(_C, _A)

    # Create serialization directory and save config in it.
    os.makedirs(_A.save_dirpath, exist_ok=True)
    _C.dump(os.path.join(_A.save_dirpath, "config.yml"))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device("cuda", _A.gpu_ids[0]) if _A.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL
    # ============================================================================================

    vocabulary = Vocabulary.from_files(_A.vocab_dirpath)
    if _A.split == "train":
        eval_dataset = QuestionCodingDataset(_A.tokens_train_h5)
    else:
        eval_dataset = QuestionCodingDataset(_A.tokens_val_h5)

    eval_dataloader = DataLoader(eval_dataset, batch_size=_C.OPTIM.BATCH_SIZE)

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_GENERATOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_GENERATOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_GENERATOR.NUM_LAYERS,
        dropout=_C.PROGRAM_GENERATOR.DROPOUT,
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=_C.QUESTION_RECONSTRUCTOR.INPUT_SIZE,
        hidden_size=_C.QUESTION_RECONSTRUCTOR.HIDDEN_SIZE,
        num_layers=_C.QUESTION_RECONSTRUCTOR.NUM_LAYERS,
        dropout=_C.QUESTION_RECONSTRUCTOR.DROPOUT,
    ).to(device)

    question_coding_checkpoint = torch.load(_A.qc_checkpoint_pthpath)
    program_generator.load_state_dict(question_coding_checkpoint["program_generator"])
    question_reconstructor.load_state_dict(question_coding_checkpoint["question_reconstructor"])

    program_generator.eval()
    question_reconstructor.eval()

    for i, batch in enumerate(tqdm(eval_dataloader)):
        for key in batch:
            batch[key] = batch[key].to(device)
        with torch.no_grad():
            # We really only need to accumulate metrics in these classes.
            _ = program_generator(batch["question"], batch["program"])
            _ = question_reconstructor(batch["program"], batch["question"])

    pg_metrics = program_generator.get_metrics()
    qr_metrics = question_reconstructor.get_metrics()

    # keys: {"BLEU", "perplexity", "sequence_accuracy"}
    for metric_name in pg_metrics:
        print(f"Program generator, {metric_name}:", pg_metrics[metric_name])
        print(f"Question reconstructor, {metric_name}:", qr_metrics[metric_name])
        print("\n")
