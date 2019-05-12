import argparse
import json
import logging
from typing import Dict, List, Union

from allennlp.data import Vocabulary
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data.datasets import JointTrainingDataset
from probnmn.models import ProgramGenerator, NeuralModuleNetwork


parser = argparse.ArgumentParser("Run inference after joint training and save model predictions.")
parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for specified phase."
)
parser.add_argument(
    "--checkpoint-path", default="", help="Path to load joint training checkpoint and evaluate."
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)

logger: logging.Logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # --------------------------------------------------------------------------------------------
    #   INPUT ARGUMENTS AND CONFIG
    # --------------------------------------------------------------------------------------------
    _A = parser.parse_args()
    _C = Config(_A.config_yml)

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Set device according to specified GPU ids.
    device = torch.device(f"cuda:{_A.gpu_ids[0]}" if _A.gpu_ids[0] >= 0 else "cpu")

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # --------------------------------------------------------------------------------------------
    #   INSTANTIATE DATALOADER AND MODELS
    # --------------------------------------------------------------------------------------------
    dataset = JointTrainingDataset(_C.DATA.TEST_TOKENS, _C.DATA.TEST_FEATURES)
    dataloader = DataLoader(dataset, batch_size=_C.OPTIM.BATCH_SIZE, num_workers=_A.cpu_workers)

    program_generator = ProgramGenerator.from_config(_C).to(device)
    nmn = NeuralModuleNetwork.from_config(_C).to(device)

    joint_training_checkpoint = torch.load(_A.checkpoint_path)
    program_generator.load_state_dict(joint_training_checkpoint["program_generator"])
    nmn.load_state_dict(joint_training_checkpoint["nmn"])

    program_generator.eval()
    nmn.eval()

    # To convert answer tokens to answer strings.
    vocabulary = Vocabulary.from_files(_C.DATA.VOCABULARY)
    predictions: List[Dict[str, Union[int, str]]] = []

    for batch in tqdm(dataloader):
        for key in batch:
            batch[key] = batch[key].to(device)

        sampled_programs = program_generator(batch["question"])["predictions"]
        answer_tokens = nmn(batch["image"], sampled_programs)["predictions"]

        for index in range(len(answer_tokens)):
            predictions.append(
                {
                    "question_index": batch["question_index"][index].item(),
                    "answer": vocabulary.get_token_from_index(
                        answer_tokens[index].item(), namespace="answers"
                    ),
                }
            )

    predictions_path = _A.checkpoint_path[:-4] + "_predictions.json"
    logger.info(f"Saving predictions to {predictions_path}")
    json.dump(predictions, open(predictions_path, "w"))
