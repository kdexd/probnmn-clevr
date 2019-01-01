import argparse
import copy
from datetime import datetime
import itertools
import json
import os

import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from tbd.data import ClevrProgramsDataset
from tbd.nn import DynamicCrossEntropyLoss
from tbd.models import ProgramPrior
import tbd.utils.checkpointing as checkpointing_utils


parser = argparse.ArgumentParser("Train program prior over CLEVR v1.0 training split programs.")
parser.add_argument(
    "-c",
    "--config-yml",
    default="configs/program_prior.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "-t",
    "--training-h5",
    default="data/training/programs.h5",
    help="Path to HDF file containing tokenized CLEVR v1.0 training split programs.",
)
parser.add_argument(
    "-v",
    "--validation-h5",
    default="data/validation/programs.h5",
    help="Path to HDF file containing tokenized CLEVR v1.0 validation split programs.",
)


parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, help="List of ids of GPUs to use (-1 for CPU)."
)
parser.add_argument(
    "--overfit", action="store_true", help="Overfit model on 5 examples, meant for debugging."
)


parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/",
    help="Path of directory to create checkpoint directory and save checkpoints.",
)
parser.add_argument(
    "--checkpoint-every",
    default=500,
    type=int,
    help="Save a checkpoint after every this many iterations.",
)


# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ================================================================================================
#   INPUT ARGUMENTS AND CONFIG
# ================================================================================================

args = parser.parse_args()

# keys: {"model", "solver"}
config = yaml.load(open(args.config_yml))
device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

# print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))


# ================================================================================================
#   SETUP DATASET, DATALOADER
# ================================================================================================

train_dataset = ClevrProgramsDataset(args.training_h5, args.overfit)
val_dataset = ClevrProgramsDataset(args.validation_h5, args.overfit)

# train dataloader can be re-initialized later while doing batch size scheduling
batch_size = config["initial_bs"]
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


# ================================================================================================
#   SETUP MODEL, CRITERION, OPTIMIZER
# ================================================================================================

program_prior = ProgramPrior(
    vocab_size=config["vocab_size"],
    embedding_size=config["embedding_size"],
    rnn_hidden_size=config["rnn_hidden_size"],
    rnn_dropout=config["rnn_dropout"],
)

# wrap around DataParallel to support multi-GPU execution
program_prior = program_prior.to(device)
if -1 not in args.gpu_ids:
    # don't wrap to DataParallel for CPU-mode
    program_prior = nn.DataParallel(program_prior, args.gpu_ids)

# declare criterion, optimizer and learning rate scheduler
criterion = DynamicCrossEntropyLoss(reduction="mean")

optimizer = optim.Adam(
    program_prior.parameters(),
    lr=config["initial_lr"],
    weight_decay=config["weight_decay"]
)

scheduler = lr_scheduler.MultiStepLR(
    optimizer,
    milestones=config["lr_steps"],
    gamma=config["lr_gamma"],
)


# ================================================================================================
#   TRAINING LOOP
# ================================================================================================

# create fresh checkpoint directory for this run
checkpoint_dirpath = checkpointing_utils.create_checkpoint_dir(args.save_dirpath, args.config_yml)
print(f"Saving checkpoints at: {checkpoint_dirpath}")

running_loss = 0.0
train_begin = datetime.now()

# keep track of best checkpoint and it's perplexity, initial with garbage right now
best_checkpoint = None
best_perplexity = 100000.0

# make train dataloader iteration cyclical
train_dataloader = itertools.cycle(train_dataloader)

for iteration in range(1, config["num_iterations"]):

    # ----------------------------------------------------------------------------------------
    #   ON ITERATION START  (shift all tensors to "device")
    # ----------------------------------------------------------------------------------------
    batch = next(train_dataloader)
    for key in batch:
        batch[key] = batch[key].to(device)

    # ----------------------------------------------------------------------------------------
    #   ITERATION: FORWARD - BACKWARD - STEP
    # ----------------------------------------------------------------------------------------
    optimizer.zero_grad()
    # shape: (batch, max_program_length, vocab_size)
    output_logits = program_prior(batch["program_tokens"], batch["program_lengths"])

    # prepare sequences for calculating cross-entropy loss
    # remove last token from logits, first token from target

    # shape: (batch, max_program_length - 1, vocab_size)
    output_logits = output_logits[:, :-1, :]
    # shape: (batch, max_program_length - 1)
    target_tokens = batch["program_tokens"][:, 1:]
    # shape: (batch, )
    target_lengths = batch["program_lengths"] - 1

    batch_loss = criterion(output_logits, target_tokens, target_lengths)
    batch_loss.backward()
    optimizer.step()

    # ----------------------------------------------------------------------------------------
    #   ON ITERATION END  (running loss, print training progress, cross val)
    # ----------------------------------------------------------------------------------------
    if running_loss > 0.0:
        running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
    else:
        running_loss = batch_loss.item()

    # print current time, iteration, running loss, learning rate
    if iteration % 100 == 0:
        print("[{}][Iter: {:6d}][Loss: {:6f}][Lr: {:6f}][Bs: {:4d}]".format(
            datetime.now() - train_begin, iteration, running_loss,
            optimizer.param_groups[0]["lr"], batch_size
        ))

    # batch size and learning rate scheduling
    scheduler.step()
    if iteration in config["bs_steps"]:
        batch_size *= config["bs_gamma"]
        train_dataloader = itertools.cycle(
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        )

    if iteration % args.checkpoint_every == 0:
        # cross-validate and report perplexity
        print(f"Cross-validation after iteration {iteration}:")
        batch_val_losses = []
        for i, batch in enumerate(tqdm(val_dataloader)):
            for key in batch:
                batch[key] = batch[key].to(device)

            with torch.no_grad():
                output_logits = program_prior(batch["program_tokens"], batch["program_lengths"])
                output_logits = output_logits[:, :-1, :]
                target_tokens = batch["program_tokens"][:, 1:]
                target_lengths = batch["program_lengths"] - 1
                batch_loss = criterion(output_logits, target_tokens, target_lengths)
                batch_val_losses.append(batch_loss)
        average_val_loss = torch.mean(torch.stack(batch_val_losses, 0)).item()
        perplexity = 2 ** average_val_loss
        print("Model perplexity: ", perplexity)

        # is this the best performing checkpoint yet?
        if perplexity < best_perplexity:
            best_perplexity = perplexity
            best_checkpoint = copy.copy(program_prior.state_dict())
        checkpointing_utils.save_checkpoint(checkpoint_dirpath, iteration, program_prior, optimizer)


# ================================================================================================
#   AFTER TRAINING ENDS
# ================================================================================================

# save best checkpoint
best_savepath = f"model_best_{best_perplexity}.pth"
torch.save(best_checkpoint, os.path.join(checkpoint_dirpath, best_savepath))
print(f"Saved best checkpoint with {best_perplexity} perplexity at {best_savepath}.")
