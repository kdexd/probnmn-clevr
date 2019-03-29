import argparse
import logging
import os
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from probnmn.config import Config
from probnmn.evaluators import ProgramPriorEvaluator, QuestionCodingEvaluator
from probnmn.trainers import ProgramPriorTrainer, QuestionCodingTrainer
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Run training for a particular phase.")
parser.add_argument(
    "--phase",
    required=True,
    choices=["program_prior", "question_coding"],
    help="Which phase to train, this argument must match 'PHASE' parameter in provided config.",
)
parser.add_argument(
    "--config-yml", required=True, help="Path to a config file for specified phase."
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)

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
    if len(_A.gpu_ids) > 1 and _C.PHASE in {"program_prior", "question_coding"}:
        logger.warning(
            f"Multi-GPU support unavailable for phase {_C.PHASE}, using only GPU {_A.gpu_ids[0]}."
        )

    # Initiliaze trainer and evaluator according to training phase.
    # TODO (kd): do something better here, this looks weird.
    trainer: Any = None
    evaluator: Any = None

    if _C.PHASE == "program_prior":
        trainer = ProgramPriorTrainer(_C, _A, device)
        evaluator = ProgramPriorEvaluator(_C, _A, trainer.models, device)
    elif _C.PHASE == "question_coding":
        trainer = QuestionCodingTrainer(_C, _A, device)
        evaluator = QuestionCodingEvaluator(_C, _A, trainer.models, device)

    for iteration in tqdm(range(_C.OPTIM.NUM_ITERATIONS), desc="training"):
        trainer.step()

        if iteration % _A.checkpoint_every == 0:
            val_metrics = evaluator.evaluate(
                num_batches=_A.num_val_examples // _C.OPTIM.BATCH_SIZE
            )
            trainer.after_validation(val_metrics)
