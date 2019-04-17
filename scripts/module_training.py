import argparse
import itertools
from typing import Dict, Optional

from allennlp.data import Vocabulary
from allennlp.nn import util as allennlp_util
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.data import ModuleTrainingDataset
from probnmn.models import ProgramGenerator
from probnmn.models.nmn import NeuralModuleNetwork
import probnmn.utils.checkpointing as checkpointing_utils
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser(
    "Train a Neural Module Network on CLEVR v1.0 using programs " "from learnt question coding."
)
parser.add_argument(
    "--config-yml",
    default="configs/module_training.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=0,
    help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--checkpoint-pthpath", default="", help="Path to load checkpoint and continue training."
)
# data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def do_iteration(batch: Dict[str, torch.Tensor],
                 program_generator: ProgramGenerator,
                 nmn: NeuralModuleNetwork,
                 optimizer: Optional[optim.Optimizer] = None):
    if nmn.training:
        optimizer.zero_grad()

    sampled_programs = program_generator(batch["question"])["predictions"]
    output_dict = nmn(batch["image"], sampled_programs, batch["answer"])

    batch_loss = output_dict["loss"].mean()

    if nmn.training:
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


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    config = common_utils.read_config(args.config_yml)
    config = common_utils.override_config_from_opts(config, args.config_override)
    common_utils.print_config_and_args(config, args)

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
        train_dataset,
        batch_size=config["optim_batch_size"],
        num_workers=args.cpu_workers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["optim_batch_size"],
        num_workers=args.cpu_workers
    )

    # Program generator will be frozen during module training.
    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["pg_input_size"],
        hidden_size=config["pg_hidden_size"],
        num_layers=config["pg_num_layers"],
        dropout=config["pg_dropout"]
    ).to(device).eval()
    program_generator.load_state_dict(torch.load(config["qc_checkpoint"])["program_generator"])

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(config["model_image_feature_size"]),
        module_channels=config["model_module_channels"],
        class_projection_channels=config["model_class_projection_channels"],
        classifier_linear_size=config["model_classifier_linear_size"]
    ).to(device)

    optimizer = optim.Adam(
        itertools.chain(program_generator.parameters(), nmn.parameters()),
        lr=config["optim_lr_initial"], weight_decay=config["optim_weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["optim_lr_gamma"],
        patience=config["optim_lr_patience"], threshold=1e-2
    )

    # Load from saved checkpoint if specified.
    if args.checkpoint_pthpath != "":
        module_training_checkpoint = torch.load(args.checkpoint_pthpath)
        nmn.load_state_dict(module_training_checkpoint["nmn"])
        optimizer.load_state_dict(module_training_checkpoint["optimizer"])
        start_iteration = int(args.checkpoint_pthpath.split("_")[-1][:-4]) + 1
    else:
        start_iteration = 1

    if -1 not in args.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        program_generator = nn.DataParallel(program_generator, args.gpu_ids)
        nmn = nn.DataParallel(nmn, args.gpu_ids)

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    checkpoint_manager = checkpointing_utils.CheckpointManager(
        serialization_dir=args.save_dirpath,
        models={"nmn": nmn},
        optimizer=optimizer,
        mode="max",
        filename_prefix="module_training",
    )
    checkpoint_manager.init_directory(config)
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)

    # Make train dataloader iteration cyclical.
    train_dataloader = common_utils.cycle(train_dataloader)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    print(f"Training for {config['optim_num_iterations']} iterations:")
    for iteration in tqdm(range(start_iteration, config["optim_num_iterations"])):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys: {"predictions", "loss", "answer_accuracy"}
        iteration_output_dict = do_iteration(batch, program_generator, nmn, optimizer)
        summary_writer.add_scalar("train/loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalar(
            "train/answer_accuracy", iteration_output_dict["answer_accuracy"], iteration
        )
        summary_writer.add_scalar(
            "train/average_invalid", iteration_output_dict["average_invalid"], iteration
        )

        # ========================================================================================
        #   VALIDATE AND (TODO) PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Validation after iteration {iteration}:")
            nmn.eval()
            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)

                with torch.no_grad():
                    iteration_output_dict = do_iteration(batch, program_generator, nmn)
                if (i + 1) * len(batch["question"]) > args.num_val_examples: break

            val_metrics = {}
            if isinstance(program_generator, nn.DataParallel):
                val_metrics["nmn"] = nmn.module.get_metrics()
            else:
                val_metrics["nmn"] = nmn.get_metrics()
            
            # Log all metrics to tensorboard.
            # For nmn, keys: {"average_invalid", answer_accuracy"}
            for model in val_metrics:
                for name in model:
                    summary_writer.add_scalar(
                        f"val/metrics/{model}/{name}", val_metrics[model][name],
                        iteration
                    )
            lr_scheduler.step(val_metrics["nmn"]["answer_accuracy"])
            checkpoint_manager.step(val_metrics["nmn"]["answer_accuracy"], iteration)
            nmn.train()
