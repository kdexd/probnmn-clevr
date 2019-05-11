import argparse
import logging
import os
from typing import Type

import numpy as np
import torch

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

# For static type hints.
from probnmn.evaluators._evaluator import _Evaluator
from probnmn.trainers._trainer import _Trainer


parser = argparse.ArgumentParser("Run training for a particular phase.")
parser.add_argument(
    "--phase",
    required=True,
    choices=["program_prior", "question_coding", "module_training", "joint_training"],
    help="Which phase to evalaute, this must match 'PHASE' parameter in provided config.",
)
parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for specified phase."
)
parser.add_argument(
    "--checkpoint-path", default="", help="Path to load checkpoint and and evaluate."
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
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml)

    # Match the phase from arguments and config parameters.
    if _A.phase != _C.PHASE:
        raise ValueError(
            f"Provided `--phase` as {_A.phase}, does not match config PHASE ({_C.PHASE})."
        )

    # Print configs and args.
    print(_C)
    for arg in vars(_A):
        print("{:<20}: {}".format(arg, getattr(_A, arg)))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    serialization_dir = os.path.dirname(_A.checkpoint_path)

    # Set trainer and evaluator classes according to phase, CamelCase for syntactic sugar.
    TrainerClass: Type[_Trainer] = (
        ProgramPriorTrainer if _C.PHASE == "program_prior" else
        QuestionCodingTrainer if _C.PHASE == "question_coding" else
        ModuleTrainingTrainer if _C.PHASE == "module_training" else
        JointTrainingTrainer
    )
    EvaluatorClass: Type[_Evaluator] = (
        ProgramPriorEvaluator if _C.PHASE == "program_prior" else
        QuestionCodingEvaluator if _C.PHASE == "question_coding" else
        ModuleTrainingEvaluator if _C.PHASE == "module_training" else
        JointTrainingEvaluator
    )
    trainer = TrainerClass(_C, serialization_dir, _A.gpu_ids, _A.cpu_workers)
    evaluator = EvaluatorClass(_C, trainer.models, _A.gpu_ids, _A.cpu_workers)

    # Load from a checkpoint to trainer for evaluation (evalautor can evaluate this checkpoint
    # because it was passed by assignment in constructor).
    trainer.load_checkpoint(_A.checkpoint_path)

    # Evalaute on full CLEVR v1.0 validation set.
    val_metrics = evaluator.evaluate()

    for model_name in val_metrics:
        for metric_name in val_metrics[model_name]:
            logger.info(
                f"val/metrics/{model_name}/{metric_name}: {val_metrics[model_name][metric_name]}"
            )
