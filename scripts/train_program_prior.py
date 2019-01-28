import argparse
from datetime import datetime
import itertools
import json
import os
import random

from allennlp.data import Vocabulary
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.data import ProgramPriorDataset
from probnmn.models import ProgramPrior
from probnmn.utils.checkpointing import CheckpointManager
import probnmn.utils.common as probnmn_utils


parser = argparse.ArgumentParser("Train program prior over CLEVR v1.0 training split programs.")
parser.add_argument(
    "--config-yml",
    default="configs/program_prior.yml",
    help="Path to a config file listing model and optimization arguments and hyperparameters.",
)
# Data file paths, gpu ids, checkpoint args etc.
probnmn_utils.add_common_args(parser)
args = parser.parse_args()

# For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def do_iteration(batch, program_prior, optimizer=None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""
    if program_prior.training:
        optimizer.zero_grad()
    # keys: {"predicted_tokens", "loss"}
    output_dict = program_prior(batch["program"])
    batch_loss = output_dict["loss"].mean()

    if program_prior.training:
        batch_loss.backward()
        # Clamp all gradients between (-5, 5).
        for parameter in program_prior.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        optimizer.step()
    return output_dict


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
    train_dataset = ProgramPriorDataset(args.tokens_train_h5)
    val_dataset = ProgramPriorDataset(args.tokens_val_h5)

    # `train_dataloader` can be re-initialized later while doing batch size scheduling
    batch_size = config["optim_bs_initial"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    program_prior = ProgramPrior(
        vocabulary,
        input_size=config["model_input_size"],
        hidden_size=config["model_hidden_size"],
        num_layers=config["model_num_layers"],
        dropout=config["model_dropout"],
    )
    optimizer = optim.Adam(
        program_prior.parameters(),
        lr=config["optim_lr_initial"],
        weight_decay=config["optim_weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["optim_lr_steps"],
        gamma=config["optim_lr_gamma"],
    )

    program_prior = program_prior.to(device)
    if -1 not in args.gpu_ids:
        # don't wrap to DataParallel for CPU-mode
        program_prior = nn.DataParallel(program_prior, args.gpu_ids)


    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)
    checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=program_prior,
        optimizer=optimizer,
        mode="min"
    )
    # Make train dataloader iteration cyclical (gets re-initialized for batch size scheduling)
    train_dataloader = probnmn_utils.cycle(train_dataloader)

    print(f"Training for {config['optim_num_iterations']}:")
    for iteration in tqdm(range(config["optim_num_iterations"])):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        output_dict = do_iteration(batch, program_prior, optimizer)

        # Batch size and learning rate scheduling
        lr_scheduler.step()
        if iteration in config["optim_bs_steps"]:
            batch_size *= config["optim_bs_gamma"]
            train_dataloader = probnmn_utils.cycle(
                DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            )
        # Log loss and schedulers to tensorboard.
        summary_writer.add_scalar("train/loss", output_dict["loss"].mean(), iteration)
        summary_writer.add_scalar("optim/learning_rate", optimizer.param_groups[0]["lr"], iteration)
        summary_writer.add_scalar("optim/batch_size", batch_size, iteration)

        # ========================================================================================
        #   VALIDATE AND PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Validation after iteration {iteration}:")
            program_prior.eval()
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    output_dict = do_iteration(batch, program_prior)

            if isinstance(program_prior, nn.DataParallel):
                program_prior_metrics = program_prior.module.get_metrics()
            else:
                program_prior_metrics = program_prior.get_metrics()

            print("\nPerplexity:", program_prior_metrics["perplexity"])
            checkpoint_manager.step(program_prior_metrics["perplexity"])
            summary_writer.add_scalars("val", program_prior_metrics, iteration)

            # Print five programs and their predicted next time-step
            print("Some predicted examples by the language program_prior (greedy decoding):")
            print("- " * 30)  # separator for neatness

            input_programs = batch["program"][:5].cpu().numpy()
            output_programs = output_dict["predicted_tokens"][:5].cpu().numpy()
            for inp, out in zip(input_programs, output_programs):
                # Print only first five time-steps, these sequences can be really long
                print("INPUT PROGRAM: ",
                      " ".join(vocabulary.get_token_from_index(i, "programs") for i in inp[:6]),
                      "...")
                # Output is one time-step shifted, but input also has a @start@ token, so the
                # shift gets cancelled.
                print("OUTPUT PROGRAM: ",
                      " ".join(vocabulary.get_token_from_index(o, "programs") for o in out[:6]),
                      "...")
                print("- " * 30)  # separator for neatness
            program_prior.train()

    # ============================================================================================
    #   AFTER TRAINING ENDS
    # ============================================================================================
    checkpoint_manager.save_best()
    print(f"Saved best checkpoint with {checkpoint_manager.metric.item()} perplexity.")
