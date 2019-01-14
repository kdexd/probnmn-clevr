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

from tbd.data import QuestionCodingDataset
from tbd.models import ProgramGenerator
from tbd.opts import add_common_opts
from tbd.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser("Question coding for CLEVR v1.0 programs and questions.")
parser.add_argument(
    "--config-yml",
    default="configs/question_coding.yml",
    help="Path to a config file listing model and solver parameters.",
)
# data file paths, gpu ids, checkpoint args etc.
add_common_opts(parser)

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def do_iteration(batch, model, optimizer=None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""
    if model.training:
        optimizer.zero_grad()
    # keys: {"predictions", "loss"}
    output_dict = model(batch["question"], batch["program"])
    batch_loss = torch.mean(output_dict["loss"])

    if model.training:
        batch_loss.backward()
        optimizer.step()
    else:
        print(output_dict["predictions"][:5])
    return batch_loss


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
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
    train_dataset = QuestionCodingDataset(args.tokens_train_h5)
    val_dataset = QuestionCodingDataset(args.tokens_val_h5)

    # train dataloader can be re-initialized later while doing batch size scheduling
    batch_size = config["initial_bs"]
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # works with 100% supervision for now
    model = ProgramGenerator(
        vocabulary, config["embedding_size"], config["rnn_hidden_size"], config["rnn_dropout"]
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=config["initial_lr"], weight_decay=config["weight_decay"]
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"],
    )

    if -1 not in args.gpu_ids:
        # don't wrap to DataParallel for CPU-mode
        model = nn.DataParallel(model, args.gpu_ids)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=model,
        optimizer=optimizer,
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

    # ============================================================================================
    #   AFTER TRAINING ENDS
    # ============================================================================================
    checkpoint_manager.save_best()
    print(f"Saved best checkpoint with {checkpoint_manager.metric.item()} perplexity.")
