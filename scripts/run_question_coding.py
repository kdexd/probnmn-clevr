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
from tbd.models import ProgramGenerator, QuestionReconstructor
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


def do_iteration(batch, program_generator, question_reconstructor, optimizer=None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""
    if program_generator.training and question_reconstructor.training:
        optimizer.zero_grad()
    # keys: {"predictions", "loss"}
    program_generator_output_dict = program_generator(batch["question"], batch["program"])

    sampled_programs = program_generator(batch["question"])["predictions"]
    if not program_generator.training:
        # pick first beam when evaluating
        # shape: (batch_size, max_decoding_steps)
        sampled_programs = sampled_programs[:, 0, :]

    question_reconstructor_output_dict = question_reconstructor(
        sampled_programs, batch["question"]
    )

    program_generation_batch_loss = torch.mean(program_generator_output_dict["loss"])
    question_reconstruction_batch_loss = torch.mean(question_reconstructor_output_dict["loss"])

    if program_generator.training and question_reconstructor.training:
        program_generation_batch_loss.backward()
        question_reconstruction_batch_loss.backward()
        optimizer.step()

    return program_generation_batch_loss, question_reconstruction_batch_loss


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
    program_generator = ProgramGenerator(
        vocabulary, config["embedding_size"], config["rnn_hidden_size"], config["rnn_dropout"]
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary, config["embedding_size"], config["rnn_hidden_size"], config["rnn_dropout"]
    ).to(device)

    optimizer = optim.Adam(
        itertools.chain(program_generator.parameters(), question_reconstructor.parameters()),
        lr=config["initial_lr"], weight_decay=config["weight_decay"]
    )
    scheduler = lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"],
    )

    if -1 not in args.gpu_ids:
        # don't wrap to DataParallel for CPU-mode
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

    pg_running_loss = 0.0
    qr_running_loss = 0.0
    train_begin = datetime.now()

    # make train dataloader iteration cyclical (gets re-initialized for batch size scheduling)
    train_dataloader = itertools.cycle(train_dataloader)

    for iteration in range(1, config["num_iterations"]):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        pg_batch_loss, qr_batch_loss = do_iteration(batch, program_generator, question_reconstructor, optimizer)

        if pg_running_loss > 0.0:
            pg_running_loss = 0.95 * pg_running_loss + 0.05 * pg_batch_loss.item()
            qr_running_loss = 0.95 * qr_running_loss + 0.05 * qr_batch_loss.item()
        else:
            pg_running_loss = pg_batch_loss.item()
            qr_running_loss = qr_batch_loss.item()

        # print current time, iteration, running loss, learning rate
        if iteration % 100 == 0:
            print("[{}][Iter: {:6d}][PG Loss: {:5f}][QR Loss: {:5f}][Lr: {:7f}][Bs: {:4d}]".format(
                datetime.now() - train_begin, iteration, pg_running_loss, qr_running_loss,
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
            program_generator.eval()
            question_reconstructor.eval()
            pg_batch_val_losses = []
            qr_batch_val_losses = []
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    pg_batch_loss, qr_batch_loss = do_iteration(batch, program_generator, question_reconstructor)
                    pg_batch_val_losses.append(pg_batch_loss)
                    qr_batch_val_losses.append(qr_batch_loss)
            print("PG BLEU:", program_generator.module.get_metrics())
            print("QR BLEU:", question_reconstructor.module.get_metrics())
            program_generator.train()
            question_reconstructor.train()

            pg_average_val_loss = torch.mean(torch.stack(pg_batch_val_losses, 0))
            qr_average_val_loss = torch.mean(torch.stack(qr_batch_val_losses, 0))
            pg_perplexity = 2 ** pg_average_val_loss
            qr_perplexity = 2 ** qr_average_val_loss
            print("PG Perplexity: ", pg_perplexity)
            print("QR Perplexity: ", qr_perplexity)

            # TODO (kd): use BLEU instead of perplexity of just program generator?
            program_generator_checkpoint_manager.step(pg_perplexity)
            question_reconstructor_checkpoint_manager.step(qr_perplexity)

    # ============================================================================================
    #   AFTER TRAINING END
    # ============================================================================================
    program_generator_checkpoint_manager.save_best()
    question_reconstructor_checkpoint_manager.save_best()
