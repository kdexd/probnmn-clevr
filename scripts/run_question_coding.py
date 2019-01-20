import argparse
import copy
from datetime import datetime
import itertools
import json
import os
import random
from typing import Dict, Optional, Union

from allennlp.data import Vocabulary
from allennlp.nn import util as allennlp_util
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from tbd.data import QuestionCodingDataset
from tbd.data.sampler import SupervisionWeightedRandomSampler
from tbd.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from tbd.utils.checkpointing import CheckpointManager
from tbd.utils.opts import add_common_opts, override_config_from_opts


parser = argparse.ArgumentParser("Question coding for CLEVR v1.0 programs and questions.")
parser.add_argument(
    "--config-yml",
    default="configs/question_coding.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--config-override",
    type=str,
    default="{}",
    help="A string following python dict syntax, specifying certain config arguments to override,"
         " useful for launching batch jobs through shel lscripts. The actual config will be "
         "updated and recorded in the checkpoint saving directory. Only argument names already "
         "present in config will be overriden, rest ignored."
)
parser.add_argument(
    "--random-seed",
    type=int,
    default=0,
    help="Random seed for all devices, useful for doing multiple runs and reporting mean/variance."
)
# data file paths, gpu ids, checkpoint args etc.
add_common_opts(parser)
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def do_iteration(config: Dict[str, Union[int, float, str]],
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

    # Sample programs from program generator, using the observed questions.
    # Sample z ~ q_\phi(z|x'), shape: (batch_size, max_program_length)
    sampled_programs = program_generator(batch["question"], greedy_decode=False)["predictions"]

    # --------------------------------------------------------------------------------------------
    # First term: question reconstruction loss (\log{p_\theta (x'|z)})
    # keys: {"predictions", "loss"}
    __qr_output_dict = question_reconstructor(sampled_programs, batch["question"])
    # shape: (batch_size, )
    negative_logprobs_reconstruction = __qr_output_dict["loss"]
    question_reconstruction_loss = torch.mean(negative_logprobs_reconstruction)
    # --------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------
    # Second term: Full monte carlo estimator of KL divergence
    # - \beta * KL (\log{q_\phi(z|x')} || \log{p(z)})
    negative_logprobs_generation = program_generator(
        batch["question"], sampled_programs, record_metrics=False)["loss"]
    negative_logprobs_prior = program_prior(sampled_programs)["loss"]

    # REINFORCE reward (R): ( \log{ (p_\theta (x'|z) * p(z) ^ \beta) / (q_\phi (z|x') ) })
    # shape: (batch_size, )
    nelbo_loss = negative_logprobs_reconstruction + \
                 config["kl_beta"] * (negative_logprobs_prior - negative_logprobs_generation)
    # All terms in previous line were negative logprobs, so negate the loss for getting reward.
    reinforce_reward = -nelbo_loss
    # Detach the reward term, we don't want gradients to flow to through that and get counted
    # twice, once already counted through path derivative loss.
    # shape: (batch_size, )
    centered_reward = (reinforce_reward - moving_average_baseline).detach()

    reinforce_loss = torch.mean(centered_reward * negative_logprobs_generation)
    path_derivative_generation_loss = torch.mean(negative_logprobs_generation)
    full_monte_carlo_kl = config["kl_beta"] * path_derivative_generation_loss + reinforce_loss

    # B := B + (1 - \delta * (R - B))
    moving_average_baseline += config["elbo_delta"] * centered_reward.mean()
    # --------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------
    # Third term: Program supervision loss: \alpha * \log{r_\phi (z'|x')}
    # keys: {"predictions", "loss"}
    __pg_output_dict = program_generator(batch["question"], batch["program"])

    # Zero out loss contribution from examples having no program supervision.
    program_supervision_loss = allennlp_util.masked_mean(
        __pg_output_dict["loss"], batch["supervision"], dim=0
    )
    # --------------------------------------------------------------------------------------------

    loss_objective = question_reconstruction_loss + config["ssl_alpha"] * program_supervision_loss
    # Baseline model does not have ELBO term. (using .get() because backward compatibility)
    if not config.get("run_baseline", True):
        loss_objective += full_monte_carlo_kl

    if program_generator.training and question_reconstructor.training:
        loss_objective.backward()
        optimizer.step()

    return {
        "predictions": {
            "__pg": __pg_output_dict["predictions"],
            "__qr": __qr_output_dict["predictions"]
        },
        "loss": {
            "reconstruction": question_reconstruction_loss,
            "full_monte_carlo_kl": full_monte_carlo_kl,
            "supervision": program_supervision_loss,
            "objective": loss_objective
        },
        "reinforce_reward": reinforce_reward.mean(),
        "baseline": moving_average_baseline
    }


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    config = yaml.load(open(args.config_yml))
    config = override_config_from_opts(config, args.config_override)

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))

    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================

    vocabulary = Vocabulary.from_files(args.vocab_dirpath)
    train_dataset = QuestionCodingDataset(
        args.tokens_train_h5, config["num_supervision"]
    )
    val_dataset = QuestionCodingDataset(args.tokens_val_h5)

    # Train dataloader and train sampler can be re-initialized later while doing batch size
    # scheduling and question max length curriculum.
    batch_size = config["initial_bs"]
    if config["do_question_curriculum"]:
        question_max_length = config["question_max_length_initial"]
    else:
        # Set arbitrary large question length to avoid curriculum.
        question_max_length = 50

    train_sampler = SupervisionWeightedRandomSampler(train_dataset, question_max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Program Prior checkpoint, this will be frozen during question coding.
    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        embedding_size=config["prior_embedding_size"],
        hidden_size=config["prior_rnn_hidden_size"],
        dropout=config["prior_rnn_dropout"]
    ).to(device)
    program_prior.load_state_dict(torch.load(config["prior_checkpoint"]))
    program_prior.eval()

    # Works with 100% supervision for now.
    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        embedding_size=config["embedding_size"],
        hidden_size=config["rnn_hidden_size"],
        dropout=config["rnn_dropout"]
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        embedding_size=config["embedding_size"],
        hidden_size=config["rnn_hidden_size"],
        dropout=config["rnn_dropout"]
    ).to(device)

    optimizer = optim.Adam(
        itertools.chain(program_generator.parameters(), question_reconstructor.parameters()),
        lr=config["initial_lr"], weight_decay=config["weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"],
    )

    if -1 not in args.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        program_prior = nn.DataParallel(program_prior, args.gpu_ids)
        program_generator = nn.DataParallel(program_generator, args.gpu_ids)
        question_reconstructor = nn.DataParallel(question_reconstructor, args.gpu_ids)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    program_generator_checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=program_generator,
        optimizer=optimizer,
        filename_prefix="program_generator",
    )
    question_reconstructor_checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=question_reconstructor,
        optimizer=optimizer,
        filename_prefix="question_reconstructor",
    )
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dirpath))

    # make train dataloader iteration cyclical (gets re-initialized for batch size scheduling)
    train_dataloader = itertools.cycle(train_dataloader)

    moving_average_baseline = torch.tensor(0.0)
    print(f"Training for {config['num_iterations']} iterations:")
    for iteration in tqdm(range(config["num_iterations"])):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        # keys: {"predictions", "loss"}
        iteration_output_dict = do_iteration(
            config, batch, program_prior, program_generator, question_reconstructor,
            moving_average_baseline, optimizer
        )
        moving_average_baseline = iteration_output_dict["baseline"]

        # Log losses and hyperparameters.
        summary_writer.add_scalars("loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars(
            "elbo", {
                "reinforce_reward": iteration_output_dict["reinforce_reward"],
                "baseline": moving_average_baseline
            },
            iteration
        )
        summary_writer.add_scalar("schedulers/batch_size", batch_size, iteration)
        summary_writer.add_scalar("schedulers/lr", optimizer.param_groups[0]["lr"], iteration)

        # Batch size and learning rate scheduling.
        if iteration in config["bs_steps"]:
            batch_size *= config["bs_gamma"]
            train_dataloader = itertools.cycle(
                DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            )
        lr_scheduler.step()

        # Curriculum training of question reconstructor based on question length.
        if config["do_question_curriculum"] and iteration in config["question_max_length_steps"]:
            question_max_length += config["question_max_length_gamma"]
            train_sampler = SupervisionWeightedRandomSampler(train_dataset, question_max_length)
            train_dataloader = itertools.cycle(
                DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
            )

        # ========================================================================================
        #   VALIDATE AND PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Cross-validation after iteration {iteration}:")
            program_generator.eval()
            question_reconstructor.eval()
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    iteration_output_dict = do_iteration(
                        config, batch, program_prior, program_generator, question_reconstructor,
                        moving_average_baseline
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
                        "program_generator": __pg_metrics[metric_name],
                        "question_reconstructor": __qr_metrics[metric_name]
                    },
                    iteration
                )
            program_generator_checkpoint_manager.step(__pg_metrics["negative_logprobs"])
            question_reconstructor_checkpoint_manager.step(__qr_metrics["negative_logprobs"])
            print("\n")
            program_generator.train()
            question_reconstructor.train()

    # ============================================================================================
    #   AFTER TRAINING END
    # ============================================================================================
    program_generator_checkpoint_manager.save_best()
    question_reconstructor_checkpoint_manager.save_best()
    summary_writer.close()
