import argparse
import os
from typing import Dict, Optional

from allennlp.data import Vocabulary
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data import ModuleTrainingDataset
from probnmn.models import ProgramGenerator
from probnmn.models.nmn import NeuralModuleNetwork
from probnmn.utils.checkpointing import CheckpointManager
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
    "--cpu-workers", type=int, default=0, help="Number of CPU workers to use for data loading."
)
parser.add_argument(
    "--checkpoint-pthpath", default="", help="Path to load checkpoint and continue training."
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)


def do_iteration(
    batch: Dict[str, torch.Tensor],
    program_generator: ProgramGenerator,
    nmn: NeuralModuleNetwork,
    optimizer: Optional[optim.Optimizer] = None,
):
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
            "metrics": {
                "answer_accuracy": output_dict["answer_accuracy"],
                "average_invalid": output_dict["average_invalid"],
            }
        }
    else:
        return_dict = {"predictions": output_dict["predictions"], "loss": batch_loss}
    return return_dict


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and _A.
    # This config object is immutable, nothing can be changed in this anymore.
    _C = Config(_A.config_yml, _A.config_override)
    common_utils.print_config_and_args(_C, _A)

    # Create serialization directory and save config in it.
    os.makedirs(_A.save_dirpath, exist_ok=True)
    _C.dump(os.path.join(_A.save_dirpath, "config.yml"))

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(_C.RANDOM_SEED)
    torch.manual_seed(_C.RANDOM_SEED)
    torch.cuda.manual_seed_all(_C.RANDOM_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # Set device according to specified GPU ids.
    device = torch.device("cuda", _A.gpu_ids[0]) if _A.gpu_ids[0] >= 0 else torch.device("cpu")

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================
    vocabulary = Vocabulary.from_files(_A.vocab_dirpath)
    train_dataset = ModuleTrainingDataset(
        _A.tokens_train_h5, _A.features_train_h5, in_memory=False
    )
    val_dataset = ModuleTrainingDataset(_A.tokens_val_h5, _A.features_val_h5, in_memory=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=_C.OPTIM.BATCH_SIZE, num_workers=_A.cpu_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=_C.OPTIM.BATCH_SIZE, num_workers=_A.cpu_workers
    )

    # Make train_dataloader cyclical to sample batches perpetually.
    train_dataloader = common_utils.cycle(train_dataloader)

    # Program generator will be frozen during module training.
    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_GENERATOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_GENERATOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_GENERATOR.NUM_LAYERS,
        dropout=_C.PROGRAM_GENERATOR.DROPOUT,
    ).to(device)

    program_generator.load_state_dict(
        torch.load(_C.CHECKPOINTS.QUESTION_CODING)["program_generator"]
    )
    program_generator.eval()

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(_C.NMN.IMAGE_FEATURE_SIZE),
        module_channels=_C.NMN.MODULE_CHANNELS,
        class_projection_channels=_C.NMN.CLASS_PROJECTION_CHANNELS,
        classifier_linear_size=_C.NMN.CLASSIFIER_LINEAR_SIZE,
    ).to(device)

    optimizer = optim.Adam(
        list(program_generator.parameters()) + list(nmn.parameters()),
        lr=_C.OPTIM.LR_INITIAL,
        weight_decay=_C.OPTIM.WEIGHT_DECAY,
    )

    # Load from saved checkpoint if specified.
    if _A.checkpoint_pthpath != "":
        module_training_checkpoint = torch.load(_A.checkpoint_pthpath)
        nmn.load_state_dict(module_training_checkpoint["nmn"])
        optimizer.load_state_dict(module_training_checkpoint["optimizer"])
        start_iteration = int(_A.checkpoint_pthpath.split("_")[-1][:-4]) + 1
    else:
        start_iteration = 1

    if -1 not in _A.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        nmn = nn.DataParallel(nmn, _A.gpu_ids)

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    summary_writer = SummaryWriter(log_dir=_A.save_dirpath)
    checkpoint_manager = CheckpointManager(
        serialization_dir=_A.save_dirpath,
        models={"nmn": nmn},
        optimizer=optimizer,
        mode="max",
        filename_prefix="module_training",
    )

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    for iteration in tqdm(range(start_iteration, _C.OPTIM.NUM_ITERATIONS), desc="training"):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys: {"predictions", "loss", "metrics"}
        iteration_output_dict = do_iteration(batch, program_generator, nmn, optimizer)
        summary_writer.add_scalar("train/loss", iteration_output_dict["loss"], iteration)
        for metric in iteration_output_dict["metrics"]:
            summary_writer.add_scalar(
                f"train/metrics/{metric}", iteration_output_dict["metrics"][metric], iteration
            )

        # ========================================================================================
        #   VALIDATE
        # ========================================================================================
        if iteration % _A.checkpoint_every == 0:
            nmn.eval()
            for i, batch in enumerate(tqdm(val_dataloader, desc="validation")):
                for key in batch:
                    batch[key] = batch[key].to(device)

                with torch.no_grad():
                    iteration_output_dict = do_iteration(batch, program_generator, nmn)
                if (i + 1) * len(batch["question"]) > _A.num_val_examples:
                    break

            val_metrics = {}
            if isinstance(nmn, nn.DataParallel):
                val_metrics["nmn"] = nmn.module.get_metrics()
            else:
                val_metrics["nmn"] = nmn.get_metrics()

            # Log all metrics to tensorboard.
            # For nmn, keys: {"average_invalid", answer_accuracy"}
            for model in val_metrics:
                for name in val_metrics[model]:
                    summary_writer.add_scalar(
                        f"val/metrics/{model}/{name}", val_metrics[model][name], iteration
                    )
            checkpoint_manager.step(val_metrics["nmn"]["answer_accuracy"], iteration)
            nmn.train()
