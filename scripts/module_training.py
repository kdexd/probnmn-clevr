import argparse
import itertools
import os
from typing import Dict, Optional
import warnings

from allennlp.data import Vocabulary
from allennlp.nn import util as allennlp_util
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from probneural_module_network.data import ModuleTrainingDataset
from probneural_module_network.data.sampler import QuestionCurriculumSampler
from probneural_module_network.models import ProgramGenerator
from probneural_module_network.models.neural_module_network import NeuralModuleNetwork
from probneural_module_network.utils.checkpointing import CheckpointManager, load_checkpoint
from probneural_module_network.utils.opts import add_common_opts, override_config_from_opts


parser = argparse.ArgumentParser(
    "Train a Neural Module Network on CLEVR v1.0 using programs " "from learnt question coding."
)
parser.add_argument(
    "--config-yml",
    default="configs/module_training.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--config-override",
    type=str,
    default="{}",
    help="A string following python dict syntax, specifying certain config arguments to override,"
    " useful for launching batch jobs through shel lscripts. The actual config will be "
    "updated and recorded in the checkpoint saving directory. Only argument names already "
    "present in config will be overriden, rest ignored.",
)
parser.add_argument(
    "--random-seed",
    type=int,
    default=0,
    help="Random seed for all devices, useful for doing multiple runs and reporting mean/variance.",
)
parser.add_argument(
    "--cpu-workers", type=int, default=16, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--checkpoint-pthpath", default="", help="Path to load checkpoint and continue training."
)
# data file paths, gpu ids, checkpoint args etc.
add_common_opts(parser)
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def do_iteration(batch: Dict[str, torch.Tensor],
                 program_generator: ProgramGenerator,
                 neural_module_network: ProbNMN,
                 optimizer: Optional[optim.Optimizer] = None):
    if neural_module_network.training:
        optimizer.zero_grad()

    sampled_programs = program_generator(batch["question"])["predictions"]
    output_dict = neural_module_network(batch["image"], sampled_programs, batch["answer"])

    batch_loss = output_dict["loss"].mean()

    if neural_module_network.training:
        batch_loss.backward()
        optimizer.step()
        return_dict = {
            "predictions": output_dict["predictions"],
            "loss": batch_loss,
            "answer_accuracy": output_dict["answer_accuracy"],
            "average_invalid": output_dict["average_invalid"],
        }
    else:
        return_dict = {
            "predictions": output_dict["predictions"],
            "loss": batch_loss
        }
    return return_dict


def cycle(iterable):
    # Using itertools.cycle with dataloader is harmful
    while True:
        for x in iterable:
            yield x


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
    train_dataset = ModuleTrainingDataset(
        args.tokens_train_h5, args.features_train_h5, in_memory=False
    )
    val_dataset = ModuleTrainingDataset(
        args.tokens_val_h5, args.features_val_h5, in_memory=False
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=args.cpu_workers
    )
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"])

    # Program generator will be frozen during module training.
    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        embedding_size=config["pg_embedding_size"],
        hidden_size=config["pg_rnn_hidden_size"],
        dropout=config["pg_rnn_dropout"],
    ).to(device)
    program_generator_model, _ = load_checkpoint(config["pg_checkpoint"])
    program_generator.load_state_dict(program_generator_model)
    program_generator.eval()

    neural_module_network = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(config["image_feature_size"]),
        module_channels=config["module_channels"],
        class_projection_channels=config["class_projection_channels"],
        classifier_linear_size=config["classifier_linear_size"],
    ).to(device)

    optimizer = optim.Adam(
        itertools.chain(program_generator.parameters(), neural_module_network.parameters()),
        lr=config["initial_lr"], weight_decay=config["weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["lr_steps"], gamma=config["lr_gamma"]
    )

    # Load from saved checkpoint if specified.
    if args.checkpoint_pthpath != "":
        neural_module_network_model, neural_module_network_optimizer = load_checkpoint(args.checkpoint_pthpath)
        neural_module_network.load_state_dict(neural_module_network_model)
        optimizer.load_state_dict(neural_module_network_optimizer)
        start_iteration = int(args.checkpoint_pthpath.split("_")[-1][:-4]) + 1
    else:
        start_iteration = 1

    if -1 not in args.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        program_generator = nn.DataParallel(program_generator, args.gpu_ids)
        neural_module_network = nn.DataParallel(neural_module_network, args.gpu_ids)

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    checkpoint_manager = CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=neural_module_network,
        optimizer=optimizer,
        mode="max",
        filename_prefix="neural_module_network",
    )
    summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dirpath))

    # Make train dataloader iteration cyclical.
    train_dataloader = cycle(train_dataloader)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    print(f"Training for {config['num_iterations']} iterations:")
    for iteration in tqdm(range(start_iteration, config["num_iterations"])):
        # Skip iteration if causes an error, there are one or two dirty batches.
        # Don't want to abruptly stop training, NMNs take time.
        try:
            batch = next(train_dataloader)
            # Shift tensors to device manually instead of looping through dict, because memory leak
            image, question = batch["image"], batch["question"]
            image, question = image.to(device), question.to(device)

            answer, program = batch["answer"], batch["program"]
            answer, program = answer.to(device), program.to(device)
            batch = {"image": image, "answer": answer, "question": question, "program": program}
            # keys: {"predictions", "loss", "answer_accuracy"}
            iteration_output_dict = do_iteration(batch, program_generator, neural_module_network, optimizer)
            summary_writer.add_scalar("train/loss", iteration_output_dict["loss"], iteration)
            summary_writer.add_scalar(
                "train/answer_accuracy", iteration_output_dict["answer_accuracy"], iteration
            )
            summary_writer.add_scalar(
                "train/average_invalid", iteration_output_dict["average_invalid"], iteration
            )
        except RuntimeError:
            warnings.warn(f"Exception thrown at {iteration} iteration during training!")
        lr_scheduler.step()

        # ========================================================================================
        #   VALIDATE AND (TODO) PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Validation after iteration {iteration}:")
            neural_module_network.eval()
            for i, batch in enumerate(tqdm(val_dataloader)):
                # Shift tensors to device manually instead of looping through ict, because memory leak (?)
                image, question = batch["image"], batch["question"]
                answer, program = batch["answer"], batch["program"]
                image, question = image.to(device), question.to(device)
                answer, program = answer.to(device), program.to(device)
                batch = {
                    "image": image,
                    "answer": answer,
                    "question": question,
                    "program": program,
                }
                with torch.no_grad():
                    iteration_output_dict = do_iteration(batch, program_generator, neural_module_network)
                if (i + 1) * len(batch["question"]) > args.num_val_examples: break

            if isinstance(program_generator, nn.DataParallel):
                val_metrics = neural_module_network.module.get_metrics()
            else:
                val_metrics = neural_module_network.get_metrics()
            answer_accuracy = val_metrics["answer_accuracy"]
            average_invalid = val_metrics["average_invalid"]

            summary_writer.add_scalar("val/answer_accuracy", answer_accuracy, iteration)
            summary_writer.add_scalar("val/average_invalid", average_invalid, iteration)
            checkpoint_manager.step(answer_accuracy, iteration)
            neural_module_network.train()

    # ============================================================================================
    #   AFTER TRAINING END
    # ============================================================================================
    checkpoint_manager.save_best()
    summary_writer.close()
