import argparse
import logging
import os

from allennlp.data import Vocabulary
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data import ProgramPriorDataset
from probnmn.models import ProgramPrior
from probnmn.utils.checkpointing import CheckpointManager
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Train program prior over CLEVR v1.0 training split programs.")
parser.add_argument(
    "--config-yml",
    default="configs/program_prior.yml",
    help="Path to a config file listing model and optimization arguments and hyperparameters.",
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)

logger: logging.Logger = logging.getLogger(__name__)


def do_iteration(batch, program_prior, optimizer=None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""
    if program_prior.training:
        optimizer.zero_grad()
    # keys: {"predictions", "loss"}
    iteration_output_dict = program_prior(batch["program"])
    batch_loss = iteration_output_dict["loss"].mean()

    if program_prior.training:
        batch_loss.backward()
        # Clamp all gradients between (-5, 5).
        for parameter in program_prior.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        optimizer.step()

    return iteration_output_dict


if __name__ == "__main__":
    # ============================================================================================
    #   INPUT ARGUMENTS AND CONFIG
    # ============================================================================================
    _A = parser.parse_args()

    # Create a config with default values, then override from config file, and args.
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
    if len(_A.gpu_ids) > 1:
        logger.warning(
            f"Multi-GPU execution not supported for training ProgramPrior because it is an "
            f"overkill, only GPU {_A.gpu_ids[0]} will be used."
        )

    # ============================================================================================
    #   SETUP VOCABULARY, DATASET, DATALOADER, MODEL, OPTIMIZER
    # ============================================================================================
    vocabulary = Vocabulary.from_files(_A.vocab_dirpath)
    train_dataset = ProgramPriorDataset(_A.tokens_train_h5)
    val_dataset = ProgramPriorDataset(_A.tokens_val_h5)

    train_dataloader = DataLoader(train_dataset, batch_size=_C.OPTIM.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=_C.OPTIM.BATCH_SIZE)

    # Make train_dataloader cyclical to sample batches perpetually.
    train_dataloader = common_utils.cycle(train_dataloader)

    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_PRIOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_PRIOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_PRIOR.NUM_LAYERS,
        dropout=_C.PROGRAM_PRIOR.DROPOUT,
    ).to(device)

    optimizer = optim.Adam(
        program_prior.parameters(), lr=_C.OPTIM.LR_INITIAL, weight_decay=_C.OPTIM.WEIGHT_DECAY
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=_C.OPTIM.LR_GAMMA,
        patience=_C.OPTIM.LR_PATIENCE,
        threshold=1e-2,
    )

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    summary_writer = SummaryWriter(log_dir=_A.save_dirpath)
    checkpoint_manager = CheckpointManager(
        serialization_dir=_A.save_dirpath,
        models={"program_prior": program_prior},
        optimizer=optimizer,
        mode="min",
        filename_prefix=_C.PHASE,
    )

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    for iteration in tqdm(range(_C.OPTIM.NUM_ITERATIONS), desc="training"):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)
        iteration_output_dict = do_iteration(batch, program_prior, optimizer)

        # Log loss and schedulers to tensorboard.
        summary_writer.add_scalar("train/loss", iteration_output_dict["loss"].mean(), iteration)

        # ========================================================================================
        #   VALIDATE AND PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % _A.checkpoint_every == 0:
            program_prior.eval()
            for i, batch in enumerate(tqdm(val_dataloader, desc="validation")):
                for key in batch:
                    batch[key] = batch[key].to(device)
                with torch.no_grad():
                    iteration_output_dict = do_iteration(batch, program_prior)

            val_metrics = {"program_prior": program_prior.get_metrics()}
            # Log all metrics to tensorboard.
            # For program prior, keys: {"perplexity"}
            for model in val_metrics:
                for name in val_metrics[model]:
                    summary_writer.add_scalar(
                        f"val/metrics/{model}/{name}", val_metrics[model][name], iteration
                    )
            checkpoint_manager.step(val_metrics["program_prior"]["perplexity"], iteration)

            # Perform learning rate scheduling (and logging) based on validation perplexity.
            lr_scheduler.step(val_metrics["program_prior"]["perplexity"])
            summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)

            # Print five programs and their predicted next time-step.
            logger.info(f"\nPredictions by program prior after iteration {iteration} (sampling):")
            logger.info("-" * 60)

            input_programs = batch["program"][:5].cpu().numpy()
            output_programs = iteration_output_dict["predictions"][:5].cpu().numpy()
            for inp, out in zip(input_programs, output_programs):
                # Print only first five time-steps, these sequences can be really long.
                input_program = " ".join(
                    vocabulary.get_token_from_index(i, "programs") for i in inp[:6]
                )
                output_program = " ".join(
                    vocabulary.get_token_from_index(o, "programs") for o in out[:6]
                )
                logger.info(f"INPUT PROGRAM: {input_program} ...")
                logger.info(f"OUTPUT PROGRAM: {output_program} ...")
                logger.info("-" * 60)

            program_prior.train()
