import argparse
import json
import os
from typing import Any, Dict, Optional
import warnings

from allennlp.data import Vocabulary
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from probnmn.data import QuestionCodingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from probnmn.modules.elbo import QuestionCodingElbo

import probnmn.utils.checkpointing as checkpointing_utils
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
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(args.random_seed)


def do_iteration(config: Dict[str, Any],
                 batch: Dict[str, torch.Tensor],
                 program_generator: ProgramGenerator,
                 question_reconstructor: QuestionReconstructor,
                 program_prior: Optional[ProgramPrior] = None,
                 elbo: Optional[QuestionCodingElbo] = None,
                 optimizer: Optional[optim.Optimizer] = None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""

    # Separate out examples with supervision and without supervision, these two lists will be
    # mutually exclusive.
    gt_supervision_indices = batch["supervision"].nonzero().squeeze()
    no_gt_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

    # --------------------------------------------------------------------------------------------
    # Supervision loss: \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} ) 

    # Pick a subset of questions without (GT) program supervision, and maximize it's
    # evidence (through a variational lower bound).
    program_tokens_gt_supervision = batch["program"][gt_supervision_indices]
    question_tokens_gt_supervision = batch["question"][gt_supervision_indices]

    # keys: {"predictions", "loss"}
    __pg_output_dict_gt_supervision = program_generator(
        question_tokens_gt_supervision, program_tokens_gt_supervision
    )
    __qr_output_dict_gt_supervision = question_reconstructor(
        program_tokens_gt_supervision, question_tokens_gt_supervision
    )

    program_generation_loss_gt_supervision = __pg_output_dict_gt_supervision["loss"].mean()
    question_reconstruction_loss_gt_supervision = __qr_output_dict_gt_supervision["loss"].mean()
    # --------------------------------------------------------------------------------------------

    if program_generator.training and question_reconstructor.training:
        optimizer.zero_grad()

        if config["qc_objective"] == "baseline":
            loss_objective = program_generation_loss_gt_supervision + \
                             question_reconstruction_loss_gt_supervision

            # Zero value variables for returning them to log on tensorboard.
            elbo_output_dict = {}
        elif config["qc_objective"] == "ours":
            # Pick a subset of questions without (GT) program supervision, and maximize it's
            # evidence (through a variational lower bound).
            question_tokens_no_gt_supervision = batch["question"][no_gt_supervision_indices]

            # keys: {"reconstruction_likelihood", "kl_divergence", "elbo", "reinforce_reward"}
            elbo_output_dict = elbo(question_tokens_no_gt_supervision)

            loss_objective = config["qc_alpha"] * (question_reconstruction_loss_gt_supervision + \
                             program_generation_loss_gt_supervision) - elbo_output_dict["elbo"]

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
            "__pg": __pg_output_dict_gt_supervision["predictions"],
            "__qr": __qr_output_dict_gt_supervision["predictions"]
        },
    }
    if program_generator.training and question_reconstructor.training:
        iteration_output_dict.update({
            "loss": {
                "question_reconstruction_gt": question_reconstruction_loss_gt_supervision,
                "program_generation_gt": program_generation_loss_gt_supervision,
            },
            "elbo": elbo_output_dict
        })
    return iteration_output_dict


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    config = common_utils.read_config(args.config_yml)
    config = common_utils.override_config_from_opts(config, args.config_override)
    common_utils.print_config_and_args(config, args)

    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")
    if len(args.gpu_ids) > 0:
        warnings.warn(f"Multi-GPU execution not supported for Question Coding because it is an"
                      f"overkill, only GPU {args.gpu_ids[0]} will be used.")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================

    vocabulary = Vocabulary.from_files(args.vocab_dirpath)
    train_dataset = QuestionCodingDataset(
        args.tokens_train_h5,
        config["qc_num_supervision"],
        supervision_question_max_length=config["qc_supervision_question_max_length"],
    )
    val_dataset = QuestionCodingDataset(args.tokens_val_h5)

    batch_size = config["optim_batch_size"]
    train_sampler = SupervisionWeightedRandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
    ).to(device)

    # Program Prior checkpoint, this will be frozen during question coding.
    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        input_size=config["prior_input_size"],
        hidden_size=config["prior_hidden_size"],
        num_layers=config["prior_num_layers"],
        dropout=config["prior_dropout"],
    ).to(device)
    program_prior.load_state_dict(torch.load(config["prior_checkpoint"])["program_prior"])
    program_prior.eval()

    elbo = QuestionCodingElbo(
        program_generator, question_reconstructor, program_prior,
        beta=config["qc_beta"],
        baseline_decay=config["qc_delta"]
    )

    optimizer = optim.Adam(
        list(program_generator.parameters()) + list(question_reconstructor.parameters()),
        lr=config["optim_lr_initial"], weight_decay=config["optim_weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["optim_lr_gamma"],
        patience=config["optim_lr_patience"], threshold=1e-2
    )

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    checkpoint_manager = checkpointing_utils.CheckpointManager(
        serialization_dir=args.save_dirpath,
        models={
            "program_generator": program_generator,
            "question_reconstructor": question_reconstructor,
        },
        optimizer=optimizer,
        mode="max",
        filename_prefix="question_coding",
    )
    checkpoint_manager.init_directory(config)
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)

    # Log a histogram of question lengths, for examples with (GT) program supervision.
    logging_utils.log_question_length_histogram(train_dataset, summary_writer)

    # Make train dataloader iteration cyclical.
    train_dataloader = common_utils.cycle(train_dataloader)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    print(f"Training for {config['optim_num_iterations']} iterations:")
    for iteration in tqdm(range(config["optim_num_iterations"])):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        # keys: {"predictions", "loss"}
        iteration_output_dict = do_iteration(
            config, batch, program_generator, question_reconstructor, program_prior, elbo, optimizer
        )
        # Log losses and hyperparameters.
        summary_writer.add_scalars("loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars("elbo", iteration_output_dict["elbo"], iteration)
        summary_writer.add_scalars("schedule", {"lr": optimizer.param_groups[0]["lr"]}, iteration)

        # ========================================================================================
        #   VALIDATE AND PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration > 0 and iteration % args.checkpoint_every == 0:
            print(f"Cross-validation after iteration {iteration}:")
            program_generator.eval()
            question_reconstructor.eval()
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    # Elbo has no role during validation, we don't need it (and hence the prior).
                    iteration_output_dict = do_iteration(
                        config, batch, program_generator, question_reconstructor
                    )
                if (i + 1) * len(batch["program"]) > args.num_val_examples: break

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
                for name in model:
                    summary_writer.add_scalar(
                        f"val/metrics/{model}/{name}", val_metrics[model][name],
                        iteration
                    )
            lr_scheduler.step(val_metrics["program_generator"]["sequence_accuracy"])
            checkpoint_manager.step(
                val_metrics["program_generator"]["sequence_accuracy"], iteration
            )
            program_generator.train()
            question_reconstructor.train()
