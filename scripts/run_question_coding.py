import argparse
import logging
import os
from typing import Any, Dict

from allennlp.data import Vocabulary
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data import QuestionCodingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from probnmn.modules.elbo import QuestionCodingElbo
from probnmn.utils.checkpointing import CheckpointManager

import probnmn.utils.common as common_utils
import probnmn.utils.logging as logging_utils


parser = argparse.ArgumentParser("Question coding for CLEVR v1.0 programs and questions.")
parser.add_argument(
    "--config-yml",
    default="configs/question_coding.yml",
    help="Path to a config file listing model and solver parameters.",
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)

logger: logging.Logger = logging.getLogger(__name__)


def do_iteration(
    config: Config,
    batch: Dict[str, torch.Tensor],
    program_generator: ProgramGenerator,
    question_reconstructor: QuestionReconstructor,
    program_prior: ProgramPrior = None,
    elbo: QuestionCodingElbo = None,
    optimizer: optim.Optimizer = None,
):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""

    # Separate out examples with supervision and without supervision, these two lists will be
    # mutually exclusive.
    supervision_indices = batch["supervision"].nonzero().squeeze()
    no_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

    # --------------------------------------------------------------------------------------------
    # Supervision loss: \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} )

    # Pick a subset of questions without (GT) program supervision, and maximize it's
    # evidence (through a variational lower bound).
    program_tokens_supervision = batch["program"][supervision_indices]
    question_tokens_supervision = batch["question"][supervision_indices]

    # keys: {"predictions", "loss"}
    __pg_output_dict_supervision = program_generator(
        question_tokens_supervision, program_tokens_supervision
    )
    __qr_output_dict_supervision = question_reconstructor(
        program_tokens_supervision, question_tokens_supervision
    )

    program_generation_loss_supervision = __pg_output_dict_supervision["loss"].mean()
    question_reconstruction_loss_supervision = __qr_output_dict_supervision["loss"].mean()
    # --------------------------------------------------------------------------------------------

    if program_generator.training and question_reconstructor.training:
        optimizer.zero_grad()

        if _C.OBJECTIVE == "baseline":
            loss_objective = (
                program_generation_loss_supervision + question_reconstruction_loss_supervision
            )

            # Zero value variables for returning them to log on tensorboard.
            elbo_output_dict: Dict[str, Any] = {}
        elif _C.OBJECTIVE == "ours":
            # Pick a subset of questions without (GT) program supervision, and maximize it's
            # evidence (through a variational lower bound).
            question_tokens_no_supervision = batch["question"][no_supervision_indices]

            # keys: {"reconstruction_likelihood", "kl_divergence", "elbo", "reinforce_reward"}
            elbo_output_dict = elbo(question_tokens_no_supervision)

            loss_objective = - elbo_output_dict["elbo"] + _C.ALPHA * (
                question_reconstruction_loss_supervision + program_generation_loss_supervision
            )

        loss_objective.backward()
        # Clamp all gradients between (-5, 5)
        for parameter in program_generator.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        for parameter in question_reconstructor.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        optimizer.step()

    iteration_output_dict = {
        "predictions": {
            # These are only needed to print examples during validation.
            "__pg": __pg_output_dict_supervision["predictions"],
            "__qr": __qr_output_dict_supervision["predictions"],
        }
    }
    if program_generator.training and question_reconstructor.training:
        iteration_output_dict.update(
            {
                "loss": {
                    "question_reconstruction_gt": question_reconstruction_loss_supervision,
                    "program_generation_gt": program_generation_loss_supervision,
                },
                "elbo": elbo_output_dict,
            }
        )
    return iteration_output_dict


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
    if len(_A.gpu_ids) > 1:
        logger.warning(
            f"Multi-GPU execution not supported for Question Coding because it is an "
            f"overkill, only GPU {_A.gpu_ids[0]} will be used."
        )

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================

    vocabulary = Vocabulary.from_files(_A.vocab_dirpath)
    train_dataset = QuestionCodingDataset(
        _A.tokens_train_h5,
        num_supervision=_C.SUPERVISION,
        supervision_question_max_length=_C.SUPERVISION_QUESTION_MAX_LENGTH,
    )
    val_dataset = QuestionCodingDataset(_A.tokens_val_h5)

    train_sampler = SupervisionWeightedRandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=_C.OPTIM.BATCH_SIZE, sampler=train_sampler
    )
    val_dataloader = DataLoader(val_dataset, batch_size=_C.OPTIM.BATCH_SIZE)

    # Make train_dataloader cyclical to sample batches perpetually.
    train_dataloader = common_utils.cycle(train_dataloader)

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

    # Program Prior checkpoint, this will be frozen during question coding.
    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_PRIOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_PRIOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_PRIOR.NUM_LAYERS,
        dropout=_C.PROGRAM_PRIOR.DROPOUT,
    ).to(device)

    program_prior.load_state_dict(torch.load(_C.CHECKPOINTS.PROGRAM_PRIOR)["program_prior"])
    program_prior.eval()

    elbo = QuestionCodingElbo(
        program_generator, question_reconstructor, program_prior, _C.BETA, _C.DELTA
    )
    optimizer = optim.Adam(
        list(program_generator.parameters()) + list(question_reconstructor.parameters()),
        lr=_C.OPTIM.LR_INITIAL,
        weight_decay=_C.OPTIM.WEIGHT_DECAY,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=_C.OPTIM.LR_GAMMA,
        patience=_C.OPTIM.LR_PATIENCE,
        threshold=1e-2,
    )

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    summary_writer = SummaryWriter(log_dir=_A.save_dirpath)
    checkpoint_manager = CheckpointManager(
        serialization_dir=_A.save_dirpath,
        models={
            "program_generator": program_generator,
            "question_reconstructor": question_reconstructor,
        },
        optimizer=optimizer,
        mode="max",
        filename_prefix="question_coding",
    )
    # Log a histogram of question lengths, for examples with (GT) program supervision.
    logging_utils.log_question_length_histogram(train_dataset, summary_writer)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    for iteration in tqdm(range(_C.OPTIM.NUM_ITERATIONS), desc="training"):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        # keys: {"predictions", "loss"}
        iteration_output_dict = do_iteration(
            _C, batch, program_generator, question_reconstructor, program_prior, elbo, optimizer
        )
        # Log losses and hyperparameters.
        summary_writer.add_scalars("train/loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars("train/elbo", iteration_output_dict["elbo"], iteration)
        summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)

        # ========================================================================================
        #   VALIDATE AND PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration > 0 and iteration % _A.checkpoint_every == 0:
            program_generator.eval()
            question_reconstructor.eval()
            for i, batch in enumerate(tqdm(val_dataloader, desc="validation")):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    # Elbo has no role during validation, we don't need it (and hence the prior).
                    iteration_output_dict = do_iteration(
                        _C, batch, program_generator, question_reconstructor
                    )
                if (i + 1) * len(batch["program"]) > _A.num_val_examples:
                    break

            # Print 10 qualitative examples from last batch.
            print(f"Qualitative examples after iteration {iteration}...\n")
            logging_utils.print_question_coding_examples(
                batch, iteration_output_dict, vocabulary, num=10
            )
            val_metrics = {
                "program_generator": program_generator.get_metrics(),
                "question_reconstructor": question_reconstructor.get_metrics(),
            }
            # Log all metrics to tensorboard.
            # For program generator, keys: {"BLEU", "perplexity", "sequence_accuracy"}
            # For question reconstructor, keys: {"BLEU", "perplexity", "sequence_accuracy"}
            for model in val_metrics:
                for name in val_metrics[model]:
                    summary_writer.add_scalar(
                        f"val/metrics/{model}/{name}", val_metrics[model][name], iteration
                    )
            lr_scheduler.step(val_metrics["program_generator"]["sequence_accuracy"])
            checkpoint_manager.step(
                val_metrics["program_generator"]["sequence_accuracy"], iteration
            )
            program_generator.train()
            question_reconstructor.train()
