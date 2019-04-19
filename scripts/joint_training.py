import argparse
import os
from typing import Dict

from allennlp.data import Vocabulary
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config
from probnmn.data import JointTrainingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from probnmn.models.nmn import NeuralModuleNetwork
from probnmn.modules.elbo import JointTrainingNegativeElbo

from probnmn.utils.checkpointing import CheckpointManager
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Jointly finetue all models usin warm starts from question "
                                 "coding/module training.")
parser.add_argument(
    "--config-yml",
    default="configs/joint_training.yml",
    help="Path to a config file listing model and solver parameters.",
)
# Data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)


def do_iteration(config: Config,
                 batch: Dict[str, torch.Tensor],
                 nmn: NeuralModuleNetwork,
                 program_generator: ProgramGenerator,
                 question_reconstructor: QuestionReconstructor,
                 program_prior: ProgramPrior,
                 elbo: JointTrainingNegativeElbo,
                 optimizer: optim.Optimizer):
    """Perform one train iteration - forward, backward passes and optim step."""
    optimizer.zero_grad()

    # Separate out examples with supervision and without supervision, these two lists will be
    # mutually exclusive.
    supervision_indices = batch["supervision"].nonzero().squeeze()
    no_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

    # Pick a subset of questions without (GT) program supervision, sample programs and pass
    # through the neural module network.
    question_tokens_no_supervision = batch["question"][no_supervision_indices]
    image_features_no_supervision = batch["image"][no_supervision_indices]
    answer_tokens_no_supervision = batch["answer"][no_supervision_indices]

    # keys: {"nmn_loss", "negative_elbo_loss"}
    elbo_output_dict = elbo(
        question_tokens_no_supervision,
        image_features_no_supervision,
        answer_tokens_no_supervision
    )

    loss_objective = _C.GAMMA * elbo_output_dict["nmn_loss"] + \
        elbo_output_dict["negative_elbo_loss"]

    if _C.OBJECTIVE == "ours":
        # ----------------------------------------------------------------------------------------
        # Supervision loss (program generator + question reconstructor):
        # Ignore question reconstructor for "baseline" objective, it's gradients don't interfere
        # with program generator anyway.

        # \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} )
        program_tokens_supervision = batch["program"][supervision_indices]
        question_tokens_supervision = batch["question"][supervision_indices]

        # keys: {"predictions", "loss"}
        __pg_output_dict_supervision = program_generator(
            question_tokens_supervision, program_tokens_supervision
        )
        __qr_output_dict_supervision = question_reconstructor(
            program_tokens_supervision, question_tokens_supervision
        )
        program_generation_loss_supervision = __pg_output_dict_supervision["loss"].mean()
        question_reconstruction_loss_supervision = __qr_output_dict_supervision["loss"].mean()
        # ----------------------------------------------------------------------------------------

        loss_objective += _C.ALPHA * (
            program_generation_loss_supervision + question_reconstruction_loss_supervision
        )

    nmn_metrics_output_dict = elbo_output_dict.pop("nmn_metrics")
    loss_objective.backward()

    # Clamp all gradients between (-5, 5)
    for parameter in list(program_generator.parameters()) + \
            list(question_reconstructor.parameters()) + \
            list(nmn.parameters()):
        if parameter.grad is not None:
            parameter.grad.clamp_(min=-5, max=5)

    optimizer.step()
    iteration_output_dict = {
        "loss": {
            "nmn_loss": elbo_output_dict["nmn_loss"]
        },
        "elbo": {
            "elbo": -elbo_output_dict["negative_elbo_loss"]
        },
        "metrics": nmn_metrics_output_dict
    }
    if _C.OBJECTIVE == "ours":
        iteration_output_dict["loss"].update({
            "question_reconstruction_gt": question_reconstruction_loss_supervision,
            "program_generation_gt": program_generation_loss_supervision,
        })
    return iteration_output_dict


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
    train_dataset = JointTrainingDataset(
        _A.tokens_train_h5,
        _A.features_train_h5,
        num_supervision=_C.SUPERVISION,
        supervision_question_max_length=_C.SUPERVISION_QUESTION_MAX_LENGTH,
    )
    val_dataset = JointTrainingDataset(
        _A.tokens_val_h5, _A.features_val_h5, num_supervision=_C.SUPERVISION
    )

    train_sampler = SupervisionWeightedRandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=_A.cpu_workers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=_C.OPTIM.BATCH_SIZE,
        num_workers=_A.cpu_workers
    )

    # Make train_dataloader cyclical to sample batches perpetually.
    train_dataloader = common_utils.cycle(train_dataloader)

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_GENERATOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_GENERATOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_GENERATOR.NUM_LAYERS,
        dropout=_C.PROGRAM_GENERATOR.DROPOUT,
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=_C.QUESTION_RECONSTRUCTOR.INPUT_SIZE,
        hidden_size=_C.QUESTION_RECONSTRUCTOR.HIDDEN_SIZE,
        num_layers=_C.QUESTION_RECONSTRUCTOR.NUM_LAYERS,
        dropout=_C.QUESTION_RECONSTRUCTOR.DROPOUT,
    ).to(device)

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(_C.NMN.IMAGE_FEATURE_SIZE),
        module_channels=_C.NMN.MODULE_CHANNELS,
        class_projection_channels=_C.NMN.CLASS_PROJECTION_CHANNELS,
        classifier_linear_size=_C.NMN.CLASSIFIER_LINEAR_SIZE,
    ).to(device)

    # Load checkpoints from question coding and module training phases.
    question_coding_checkpoint = torch.load(_C.CHECKPOINTS.QUESTION_CODING)
    program_generator.load_state_dict(question_coding_checkpoint["program_generator"])
    question_reconstructor.load_state_dict(question_coding_checkpoint["question_reconstructor"])
    nmn.load_state_dict(torch.load(_C.CHECKPOINTS.NMN)["nmn"])

    # ProgramPrior checkpoint, this will be frozen during joint training.
    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        input_size=_C.PROGRAM_PRIOR.INPUT_SIZE,
        hidden_size=_C.PROGRAM_PRIOR.HIDDEN_SIZE,
        num_layers=_C.PROGRAM_PRIOR.NUM_LAYERS,
        dropout=_C.PROGRAM_PRIOR.DROPOUT,
    ).to(device)

    program_prior.load_state_dict(torch.load(_C.CHECKPOINTS.PROGRAM_PRIOR)["program_prior"])
    program_prior.eval()

    elbo = JointTrainingNegativeElbo(
        program_generator, question_reconstructor, program_prior, nmn,
        beta=_C.BETA, gamma=_C.GAMMA, baseline_decay=_C.DELTA,
        objective=_C.OBJECTIVE
    )

    all_parameters = (
        list(program_generator.parameters()) + 
        list(question_reconstructor.parameters()) + 
        list(nmn.parameters())
    )
    optimizer = optim.Adam(
        all_parameters,
        lr=_C.OPTIM.LR_INITIAL,
        weight_decay=_C.OPTIM.WEIGHT_DECAY,
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=_C.OPTIM.LR_GAMMA,
        patience=_C.OPTIM.LR_PATIENCE,
        threshold=1e-2,
    )

    if -1 not in _A.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        nmn = nn.DataParallel(nmn, _A.gpu_ids)

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    summary_writer = SummaryWriter(log_dir=_A.save_dirpath)
    checkpoint_manager = CheckpointManager(
        serialization_dir=_A.save_dirpath,
        models={
            "program_generator": program_generator,
            "question_reconstructor": question_reconstructor,
            "nmn": nmn,
        },
        optimizer=optimizer,
        mode="max",
        filename_prefix="joint_training",
    )

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    for iteration in tqdm(range(_C.OPTIM.NUM_ITERATIONS), desc="training"):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys: {"predictions", "loss", "metrics"}
        iteration_output_dict = do_iteration(
            _C, batch, nmn, program_generator, question_reconstructor, program_prior,
            elbo, optimizer
        )

        # Log losses and hyperparameters.
        summary_writer.add_scalars("train/loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars("train/elbo", iteration_output_dict["elbo"], iteration)
        summary_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
        for metric in iteration_output_dict["metrics"]:
            summary_writer.add_scalar(
                f"train/metrics/{metric}", iteration_output_dict["metrics"][metric], iteration
            )

        # ========================================================================================
        #   VALIDATION
        # ========================================================================================
        if iteration % _A.checkpoint_every == 0:
            nmn.eval()
            program_generator.eval()
            question_reconstructor.eval()

            for i, batch in enumerate(tqdm(val_dataloader, desc="validation")):
                for key in batch:
                    batch[key] = batch[key].to(device)

                # Just accumulate metrics across batches, in these models, by a forward pass.
                with torch.no_grad():
                    sampled_programs = program_generator(
                        batch["question"], batch["program"])["predictions"]
                    _ = question_reconstructor(batch["program"], batch["question"])
                    _ = nmn(batch["image"], sampled_programs, batch["answer"])

                if (i + 1) * len(batch["question"]) > _A.num_val_examples: break

            val_metrics = {
                "program_generator": program_generator.get_metrics(),
                "question_reconstructor": question_reconstructor.get_metrics(),
                "nmn": nmn.module.get_metrics() if isinstance(nmn, nn.DataParallel)
                else nmn.get_metrics()
            }

            # Log all metrics to tensorboard.
            # For program generator, keys: {"BLEU", "perplexity", "sequence_accuracy"}
            # For question reconstructor, keys: {"BLEU", "perplexity", "sequence_accuracy"}
            # For nmn, keys: {"average_invalid", answer_accuracy"}
            for model in val_metrics:
                for name in val_metrics[model]:
                    summary_writer.add_scalar(
                        f"val/metrics/{model}/{name}", val_metrics[model][name],
                        iteration
                    )
            lr_scheduler.step(val_metrics["nmn"]["answer_accuracy"])
            checkpoint_manager.step(val_metrics["nmn"]["sequence_accuracy"], iteration)

            program_generator.train()
            question_reconstructor.train()
            nmn.train()
