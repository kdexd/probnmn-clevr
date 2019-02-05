import argparse
import itertools
import os
from typing import Dict, Optional, Union

from allennlp.data import Vocabulary
from allennlp.nn import util as allennlp_util
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
from probnmn.utils.checkpointing import CheckpointManager, load_checkpoint
import probnmn.utils.common as probnmn_utils


parser = argparse.ArgumentParser("Question coding for CLEVR v1.0 programs and questions.")
parser.add_argument(
    "--config-yml",
    default="configs/question_coding.yml",
    help="Path to a config file listing model and solver parameters.",
)
# Data file paths, gpu ids, checkpoint args etc.
probnmn_utils.add_common_args(parser)
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
np.random.seed(args.random_seed)


def do_iteration(iteration: int,
                 config: Dict[str, Union[int, float, str]],
                 batch: Dict[str, torch.Tensor],
                 program_prior: ProgramPrior,
                 program_generator: ProgramGenerator,
                 question_reconstructor: QuestionReconstructor,
                 moving_average_baseline: torch.Tensor,
                 optimizer: Optional[optim.Optimizer] = None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""

    # Notations in comments: x', z' are questions and programs observed in dataset.
    # z are sampled programs given x' from the dataset.
    # In other words x' = batch["questions"] and z' = batch["programs"]
    if program_generator.training and question_reconstructor.training:
        optimizer.zero_grad()

    # Separate out examples with supervision and without supervision, these two maskes will be
    # mutually exclusive. Shape: (batch_size, 1)
    gt_supervision_mask = batch["supervision"]
    no_gt_supervision_mask = 1 - gt_supervision_mask

    # --------------------------------------------------------------------------------------------
    # Supervision loss: \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} ) 
    # keys: {"predictions", "loss", "sequence_logprobs"}
    __pg_output_dict_gt_supervision = program_generator(batch["question"], batch["program"])
    __qr_output_dict_gt_supervision = question_reconstructor(batch["program"], batch["question"])
    # Zero out loss contribution from examples having no program supervision.
    program_generation_loss_gt_supervision = allennlp_util.masked_mean(
        __pg_output_dict_gt_supervision["loss"], gt_supervision_mask, dim=-1
    )
    question_reconstruction_loss_gt_supervision = allennlp_util.masked_mean(
        __qr_output_dict_gt_supervision["loss"], gt_supervision_mask, dim=-1
    )
    # --------------------------------------------------------------------------------------------

    if config["qc_objective"] == "baseline":
        loss_objective = program_generation_loss_gt_supervision + \
                         question_reconstruction_loss_gt_supervision

        # Zero value variables for returning them to log on tensorboard.
        question_reconstruction_loss_no_gt_supervision = 0
        full_monte_carlo_kl = 0
        reward = 0
        moving_average_baseline = 0
    elif config["qc_objective"] == "ours":
        # ----------------------------------------------------------------------------------------
        # Full monte carlo gradient estimator
        # \log{p_\theta (x'|z)} - \beta * KL (\log{q_\phi(z|x')} || \log{p(z)})

        # Everything here will be calculated for examples with no (GT) supervision, examples with
        # GT supervision only need to maximize evidence.

        # Sample programs, for the questions of examples with no GT (program) supervision.
        # Sample z ~ q_\phi(z|x'), shape: (batch_size, max_program_length)

        # keys: {"predictions", "loss", "sequence_logprobs"}
        __pg_output_dict_no_gt_supervision = program_generator(batch["question"])
        sampled_programs = __pg_output_dict_no_gt_supervision["predictions"]

        # keys: {"predictions", "loss", "sequence_logprobs"}
        __qr_output_dict_no_gt_supervision = question_reconstructor(
            sampled_programs, batch["question"], record_metrics=False
        )

        question_reconstruction_loss_no_gt_supervision = allennlp_util.masked_mean(
            __qr_output_dict_no_gt_supervision["loss"],
            no_gt_supervision_mask, dim=-1
        )

        # Mask all of these while computing net REINFORCE loss.
        # shape: (batch_size, )
        logprobs_reconstruction = __qr_output_dict_no_gt_supervision["sequence_logprobs"]
        logprobs_generation = __pg_output_dict_no_gt_supervision["sequence_logprobs"]
        logprobs_prior = program_prior(sampled_programs)["sequence_logprobs"]

        # REINFORCE reward (R): ( \log{ (p_\theta (x'|z) * p(z) ^ \beta) / (q_\phi (z|x') ) })
        # shape: (batch_size, )
        reinforce_reward = logprobs_reconstruction + \
                           config["qc_beta"] * (logprobs_prior - logprobs_generation)

        # Detach the reward term, we don't want gradients to flow to through that and get counted
        # twice, once already counted through path derivative loss.
        reinforce_reward = reinforce_reward.detach()
        # Zero out the net reward for examples with GT (program) supervision.
        centered_reward = (reinforce_reward - moving_average_baseline) * \
                          no_gt_supervision_mask.float()

        # Set reward for examples with GT (program) supervision to zero.
        reinforce_loss = - allennlp_util.masked_mean(
            centered_reward * logprobs_generation,
            no_gt_supervision_mask,
            dim=-1
        )
        path_derivative_generation_loss = - allennlp_util.masked_mean(
            logprobs_generation, no_gt_supervision_mask, dim=-1
        )
        full_monte_carlo_kl = question_reconstruction_loss_no_gt_supervision + \
                              config["qc_beta"] * path_derivative_generation_loss + \
                              reinforce_loss

        # B := B + (1 - \delta * (R - B))
        moving_average_baseline += config["qc_delta"] * allennlp_util.masked_mean(
            centered_reward, no_gt_supervision_mask, dim=-1
        )
        reward = allennlp_util.masked_mean(reinforce_reward, no_gt_supervision_mask, dim=-1)
        # ----------------------------------------------------------------------------------------

        loss_objective = config["qc_alpha"] * question_reconstruction_loss_gt_supervision + \
                         config["qc_alpha"] * program_generation_loss_gt_supervision + \
                         full_monte_carlo_kl

    if program_generator.training and question_reconstructor.training:
        loss_objective.backward()
        # Clamp all gradients between (-5, 5)
        for parameter in program_generator.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        for parameter in question_reconstructor.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        optimizer.step()

    return {
        "predictions": {
            # These only matter during validation.
            "__pg": __pg_output_dict_gt_supervision["predictions"],
            "__qr": __qr_output_dict_gt_supervision["predictions"]
        },
        "loss": {
            "question_reconstruction_no_gt": question_reconstruction_loss_no_gt_supervision,
            "question_reconstruction_gt": question_reconstruction_loss_gt_supervision,
            "full_monte_carlo_kl": full_monte_carlo_kl,
            "program_generation_gt": program_generation_loss_gt_supervision,
            "objective": loss_objective
        },
        "reward": reward,
        "baseline": moving_average_baseline
    }


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    config = probnmn_utils.read_config(args.config_yml)
    config = probnmn_utils.override_config_from_opts(config, args.config_override)
    probnmn_utils.print_config_and_args(config, args)

    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================

    vocabulary = Vocabulary.from_files(args.vocab_dirpath)
    train_dataset = QuestionCodingDataset(
        args.tokens_train_h5, config["qc_num_supervision"]
    )
    val_dataset = QuestionCodingDataset(args.tokens_val_h5)

    batch_size = config["optim_batch_size"]
    train_sampler = SupervisionWeightedRandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Program Prior checkpoint, this will be frozen during question coding.
    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        input_size=config["prior_input_size"],
        hidden_size=config["prior_hidden_size"],
        num_layers=config["prior_num_layers"],
        dropout=config["prior_dropout"],
        average_loss_across_timesteps=config["qc_average_loss_across_timesteps"],
        average_logprobs_across_timesteps=config["qc_average_logprobs_across_timesteps"],
    ).to(device)
    prior_model, prior_optimizer = load_checkpoint(config["prior_checkpoint"])
    program_prior.load_state_dict(prior_model)
    program_prior.eval()

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
        average_loss_across_timesteps=config["qc_average_loss_across_timesteps"],
        average_logprobs_across_timesteps=config["qc_average_logprobs_across_timesteps"],
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
        average_loss_across_timesteps=config["qc_average_loss_across_timesteps"],
        average_logprobs_across_timesteps=config["qc_average_logprobs_across_timesteps"],
    ).to(device)

    optimizer = optim.Adam(
        itertools.chain(program_generator.parameters(), question_reconstructor.parameters()),
        lr=config["optim_lr_initial"], weight_decay=config["optim_weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["optim_lr_gamma"],
        patience=config["optim_lr_patience"], threshold=1e-2
    )

    if -1 not in args.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        program_prior = nn.DataParallel(program_prior, args.gpu_ids)
        program_generator = nn.DataParallel(program_generator, args.gpu_ids)
        question_reconstructor = nn.DataParallel(question_reconstructor, args.gpu_ids)

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    program_generator_checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=program_generator,
        optimizer=optimizer,
        filename_prefix="program_generator",
        mode="min",
    )
    question_reconstructor_checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=question_reconstructor,
        optimizer=optimizer,
        filename_prefix="question_reconstructor",
        mode="min",
    )
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dirpath))

    # make train dataloader iteration cyclical (gets re-initialized for batch size scheduling)
    train_dataloader = probnmn_utils.cycle(train_dataloader)

    moving_average_baseline = torch.tensor(0.0)

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
            iteration, config, batch, program_prior, program_generator, question_reconstructor,
            moving_average_baseline, optimizer
        )
        moving_average_baseline = iteration_output_dict["baseline"]

        # Log losses and hyperparameters.
        summary_writer.add_scalars("loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars(
            "elbo", {
                "reward": iteration_output_dict["reward"],
                "baseline": moving_average_baseline
            },
            iteration
        )
        summary_writer.add_scalar("schedulers/lr", optimizer.param_groups[0]["lr"], iteration)

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
                    iteration_output_dict = do_iteration(
                        iteration, config, batch, program_prior, program_generator,
                        question_reconstructor, moving_average_baseline
                    )
                if (i + 1) * len(batch["program"]) > args.num_val_examples: break

            # Print 10 qualitative examples from last batch.
            print(f"Qualitative examples after iteration {iteration}...")
            print("- " * 30)
            for j in range(min(len(batch["question"]), 20)):
                print("PROGRAM: " + " ".join(
                    [vocabulary.get_token_from_index(p_index.item(), "programs")
                     for p_index in batch["program"][j] if p_index != 0]
                ))

                print("SAMPLED PROGRAM: " + " ".join(
                    [vocabulary.get_token_from_index(p_index.item(), "programs")
                     for p_index in iteration_output_dict["predictions"]["__pg"][j]
                     if p_index != 0]
                ))

                print("QUESTION: " + " ".join(
                    [vocabulary.get_token_from_index(q_index.item(), "questions")
                     for q_index in batch["question"][j] if q_index != 0]
                ))

                print("RECONST QUESTION: " + " ".join(
                    [vocabulary.get_token_from_index(q_index.item(), "questions")
                     for q_index in iteration_output_dict["predictions"]["__qr"][j]
                     if q_index != 0]
                ))
                print("- " * 30)

            # Log BLEU score and perplexity of both program_generator and question_reconstructor.
            if isinstance(program_generator, nn.DataParallel):
                __pg_metrics = program_generator.module.get_metrics()
                __qr_metrics = question_reconstructor.module.get_metrics()
            else:
                __pg_metrics = program_generator.get_metrics()
                __qr_metrics = question_reconstructor.get_metrics()

            # Log three metrics to tensorboard.
            # keys: {"BLEU", "perplexity", "sequence_accuracy"}
            for metric_name in __pg_metrics:
                summary_writer.add_scalars(
                    "metrics/" + metric_name, {
                        "program_generation": __pg_metrics[metric_name],
                        "question_reconstruction": __qr_metrics[metric_name]
                    },
                    iteration
                )

            # Learning rate scheduling.
            lr_scheduler.step(__pg_metrics["sequence_accuracy"])

            program_generator_checkpoint_manager.step(__pg_metrics["perplexity"], iteration)
            question_reconstructor_checkpoint_manager.step(__qr_metrics["perplexity"], iteration)
            print("\n")
            program_generator.train()
            question_reconstructor.train()

    # ============================================================================================
    #   AFTER TRAINING END
    # ============================================================================================
    program_generator_checkpoint_manager.save_best()
    question_reconstructor_checkpoint_manager.save_best()
    summary_writer.close()
