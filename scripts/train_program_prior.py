import argparse
import copy
from datetime import datetime
import itertools
import json
import os
import random

from allennlp.data import Vocabulary
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from tbd.data import ClevrProgramsDataset
from tbd.models import ProgramPrior
from tbd.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser("Train program prior over CLEVR v1.0 training split programs.")
parser.add_argument(
    "--config-yml",
    default="configs/program_prior.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--training-h5",
    default="data/training/programs.h5",
    help="Path to HDF file containing tokenized CLEVR v1.0 training split programs.",
)
parser.add_argument(
    "--validation-h5",
    default="data/validation/programs.h5",
    help="Path to HDF file containing tokenized CLEVR v1.0 validation split programs.",
)
parser.add_argument(
    "--vocab-dirpath",
    default="data/clevr_vocab",
    help="Path to directory containing vocabulary for programs, questions and answers.",
)


parser.add_argument_group("Arguments independent of experiment reproducibility")
parser.add_argument(
    "--gpu-ids", nargs="+", type=int, help="List of ids of GPUs to use (-1 for CPU)."
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="checkpoints/program_prior",
    help="Path of directory to save checkpoints, this path is recommended to be empty or "
         "non-existent. Having previously saved checkpoints in this directory might "
         "overwrite them.",
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


def do_iteration(batch, model, optimizer=None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""
    if model.training:
        optimizer.zero_grad()
    # keys: {"predicted_tokens", "loss"}
    output_dict = model(batch["program_tokens"])
    batch_loss = torch.mean(output_dict["loss"])

    if model.training:
        batch_loss.backward()
        optimizer.step()
    return batch_loss


if __name__ == "__main__":
    # ================================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ================================================================================================
    args = parser.parse_args()
    config = yaml.load(open(args.config_yml))
    device = torch.device("cuda", args.gpu_ids[0]) if args.gpu_ids[0] >= 0 else torch.device("cpu")

    # print config and args
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print("{:<20}: {}".format(arg, getattr(args, arg)))

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================

    vocabulary = Vocabulary.from_files(args.vocab_dirpath)
    train_dataset = ClevrProgramsDataset(args.training_h5)
    val_dataset = ClevrProgramsDataset(args.validation_h5)

    # train dataloader can be re-initialized later while doing batch size scheduling
    batch_size = config["initial_bs"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    model = ProgramPrior(
        vocabulary,
        embedding_size=config["embedding_size"],
        hidden_size=config["rnn_hidden_size"],
        dropout=config["rnn_dropout"],
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=config["initial_lr"],
        weight_decay=config["weight_decay"]
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["lr_steps"],
        gamma=config["lr_gamma"],
    )

    model = model.to(device)
    if -1 not in args.gpu_ids:
        # don't wrap to DataParallel for CPU-mode
        model = nn.DataParallel(model, args.gpu_ids)


    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config_ymlpath=args.config_yml,
        model=model,
        optimizer=optimizer,
        metric_mode="min"
    )

    running_loss = 0.0
    train_begin = datetime.now()

    # make train dataloader iteration cyclical (gets re-initialized for batch size scheduling)
    train_dataloader = itertools.cycle(train_dataloader)

    for iteration in range(1, config["num_iterations"]):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        batch_loss = do_iteration(batch, model, optimizer)

        if running_loss > 0.0:
            running_loss = 0.95 * running_loss + 0.05 * batch_loss.item()
        else:
            running_loss = batch_loss.item()

        # print current time, iteration, running loss, learning rate
        if iteration % 100 == 0:
            print("[{}][Iter: {:6d}][Loss: {:5f}][Lr: {:7f}][Bs: {:4d}]".format(
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

        # ========================================================================================
        #   VALIDATE AND PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Cross-validation after iteration {iteration}:")
            model.eval()
            batch_val_losses = []
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    batch_loss = do_iteration(batch, model)
                    batch_val_losses.append(batch_loss)
            model.train()

            average_val_loss = torch.mean(torch.stack(batch_val_losses, 0))
            perplexity = 2 ** average_val_loss
            print("Model perplexity: ", perplexity.item())
            checkpoint_manager.step(perplexity)

            # print three random programs and their outputs
            # print("Some predicted examples by the language model (greedy decoding):")
            # print_dataloader = DataLoader(val_dataset, batch_size=3, shuffle=True)
            # batch = next(iter(print_dataloader))
            # with torch.no_grad():
            #     output_dict = model(batch["program_tokens"], batch["program_lengths"])
            #     _, output_tokens = torch.max(output_logits, dim=-1)

            # input_programs = batch["program_tokens"].cpu().numpy()
            # output_programs = output_tokens.cpu().numpy()
            # for inp, out in zip(input_programs, output_programs):
            #     print("INPUT PROGRAM: ", " ".join(vocabulary.to_words(inp)[1:6]), "...")
            #     print("OUTPUT PROGRAM:", " ".join(vocabulary.to_words(out)[0:5]), "...", "\n")

    # ============================================================================================
    #   AFTER TRAINING ENDS
    # ============================================================================================
    checkpoint_manager.save_best()
    print(f"Saved best checkpoint with {checkpoint_manager.metric.item()} perplexity.")
