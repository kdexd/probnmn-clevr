import argparse
import logging
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from probnmn.config import Config
from probnmn.evaluators import (
    ProgramPriorEvaluator,
    JointTrainingEvaluator,
    ModuleTrainingEvaluator,
    QuestionCodingEvaluator,
)
from probnmn.trainers import (
    ProgramPriorTrainer,
    JointTrainingTrainer,
    ModuleTrainingTrainer,
    QuestionCodingTrainer,
)


parser = argparse.ArgumentParser("Run training for a particular phase.")
parser.add_argument(
    "--phase",
    required=True,
    choices=["program_prior", "question_coding", "module_training", "joint_training"],
    help="Which phase to train, this argument must match 'PHASE' parameter in provided config.",
)
parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for specified phase."
)
parser.add_argument(
    "--config-override",
    default=[],
    nargs="*",
    help="A sequence of key-value pairs specifying certain config arguments (with dict-like "
    "nesting) using a dot operator. The actual config will be updated and recorded in "
    "the serialization directory.",
)

parser.add_argument_group("Compute resource management arguments.")
parser.add_argument(
    "--gpu-ids", required=True, nargs="+", type=int, help="List of GPU IDs to use (-1 for CPU)."
)
parser.add_argument(
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)

parser.add_argument_group("Checkpointing related arguments.")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/experiment",
    help="Path to a (non-existent) directory for serializing checkpoints and tensorboard logs.",
)
parser.add_argument(
    "--checkpoint-every",
    default=500,
    type=int,
    help="Save a checkpoint after every this many epochs/iterations.",
)
parser.add_argument(
    "--start-from-checkpoint",
    default="",
    help="Path to load checkpoint and continue training [only supported for module_training].",
)
parser.add_argument(
    "--num-val-batches",
    default=256,
    type=int,
    help="Number of batches to validate on - can be used for fast validation, although might "
    "provide a noisy estimate of performance.",
)

logger: logging.Logger = logging.getLogger(__name__)


if __name__ == "__main__":
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)

    # Match the phase from arguments and config parameters.
    if _A.phase != _C.PHASE:
        raise ValueError(
            f"Provided `--phase` as {_A.phase}, does not match config PHASE ({_C.PHASE})."
        )

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # Create serialization directory and save config in it.
    os.makedirs(_A.serialization_dir, exist_ok=True)
    _C.dump(os.path.join(_A.serialization_dir, "config.yml"))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device("cuda", _A.gpu_ids[0]) if _A.gpu_ids[0] >= 0 else torch.device("cpu")
    if len(_A.gpu_ids) > 1 and _C.PHASE in {"program_prior", "question_coding"}:
        logger.warning(
            f"Multi-GPU support unavailable for phase {_C.PHASE}, using only GPU {_A.gpu_ids[0]}."
        )

    # Initialize trainer and evaluator according to training phase.
    # TODO (kd): do something better here, this looks weird.
    trainer: Any = None
    evaluator: Any = None

    if _C.PHASE == "program_prior":
        trainer = ProgramPriorTrainer(_C, _A.serialization_dir, _A.gpu_ids)
        evaluator = ProgramPriorEvaluator(_C, trainer.models, device)
    elif _C.PHASE == "question_coding":
        trainer = QuestionCodingTrainer(_C, _A.serialization_dir, _A.gpu_ids)
        evaluator = QuestionCodingEvaluator(_C, trainer.models, _A.gpu_ids)
    elif _C.PHASE == "module_training":
        trainer = ModuleTrainingTrainer(_C, _A.serialization_dir, _A.gpu_ids)
        evaluator = ModuleTrainingEvaluator(_C, trainer.models, _A.gpu_ids)
    elif _C.PHASE == "joint_training":
        trainer = JointTrainingTrainer(_C, _A.serialization_dir, _A.gpu_ids)
        evaluator = JointTrainingEvaluator(_C, trainer.models, _A.gpu_ids)

    # Load from a checkpoint if specified, and resume training from there.
    if _A.start_from_checkpoint != "":
        trainer.load_iteration(_A.start_from_checkpoint)

    for iteration in tqdm(range(_C.OPTIM.NUM_ITERATIONS), desc="training"):
        trainer.step()

        if iteration % _A.checkpoint_every == 0:
            val_metrics = evaluator.evaluate(num_batches=_A.num_val_batches)
            trainer.after_validation(val_metrics)
