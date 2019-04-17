import argparse
import itertools
from typing import Any, Dict

from allennlp.data import Vocabulary
from allennlp.nn import util as allennlp_util
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.data import JointTrainingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from probnmn.models.nmn import NeuralModuleNetwork
from probnmn.modules.elbo import JointTrainingNegativeElbo, Reinforce

import probnmn.utils.checkpointing as checkpointing_utils
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Jointly finetue all models usin warm starts from question "
                                 "coding/module training.")
parser.add_argument(
    "--config-yml",
    default="configs/joint_training.yml",
    help="Path to a config file listing model and solver parameters.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=0,
    help="Number of CPU workers to use for data loading."
)
# data file paths, gpu ids, checkpoint args etc.
common_utils.add_common_args(parser)
args = parser.parse_args()

# for reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def do_iteration(config: Dict[str, Any],
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
    gt_supervision_indices = batch["supervision"].nonzero().squeeze()
    no_gt_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

    # Pick a subset of questions without (GT) program supervision, sample programs and pass
    # through the neural module network.
    question_tokens_no_gt_supervision = batch["question"][no_gt_supervision_indices]
    image_features_no_gt_supervision = batch["image"][no_gt_supervision_indices]
    answer_tokens_no_gt_supervision = batch["answer"][no_gt_supervision_indices]

    # keys: {"nmn_loss", "negative_elbo_loss"}
    elbo_output_dict = elbo(
        question_tokens_no_gt_supervision,
        image_features_no_gt_supervision,
        answer_tokens_no_gt_supervision
    )

    loss_objective = config["jt_gamma"] * elbo_output_dict["nmn_loss"] + \
                     elbo_output_dict["negative_elbo_loss"]

    if config["jt_objective"] == "ours":
        # ----------------------------------------------------------------------------------------
        # Supervision loss (program generator + question reconstructor):
        # Ignore question reconstructor for "baseline" objective, it's gradients don't interfere
        # with program generator anyway.

        # \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} )
        program_tokens_gt_supervision = batch["program"][gt_supervision_indices]
        question_tokens_gt_supervision = batch["question"][gt_supervision_indices]

        # keys: {"predictions", "loss"}
        __pg_output_dict_gt_supervision = program_generator(
            question_tokens_gt_supervision, program_tokens_gt_supervision
        )
        __qr_output_dict_gt_supervision = question_reconstructor(
            program_tokens_gt_supervision, question_tokens_gt_supervision
        )
        program_generation_loss_gt_supervision = __pg_output_dict_gt_supervision["loss"].mean()
        question_reconstruction_loss_gt_supervision = __qr_output_dict_gt_supervision["loss"].mean()
        # ----------------------------------------------------------------------------------------

        loss_objective += config["qc_alpha"] * program_generation_loss_gt_supervision + \
                          config["qc_alpha"] * question_reconstruction_loss_gt_supervision

    nmn_metrics_output_dict = elbo_output_dict.pop("nmn_metrics")
    loss_objective.backward()

    # Clamp all gradients between (-5, 5)
    for parameter in itertools.chain(program_generator.parameters(),
                                     question_reconstructor.parameters(),
                                     nmn.parameters()):
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
        "nmn_metrics": nmn_metrics_output_dict
    }
    if config["jt_objective"] == "ours":
        iteration_output_dict["loss"].update({
            "question_reconstruction_gt": question_reconstruction_loss_gt_supervision,
            "program_generation_gt": program_generation_loss_gt_supervision,
        })
    return iteration_output_dict


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
    train_dataset = JointTrainingDataset(
        args.tokens_train_h5,
        args.features_train_h5,
        num_supervision=config["qc_num_supervision"],
        supervision_question_max_length=config["qc_supervision_question_max_length"],
    )
    val_dataset = JointTrainingDataset(
        args.tokens_val_h5, args.features_val_h5, num_supervision=config["qc_num_supervision"]
    )

    train_sampler = SupervisionWeightedRandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["optim_batch_size"],
        sampler=train_sampler,
        num_workers=args.cpu_workers
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["optim_batch_size"],
        num_workers=args.cpu_workers
    )
    # Program Prior checkpoint, this will be frozen during question coding.
    program_prior = ProgramPrior(
        vocabulary=vocabulary,
        input_size=config["prior_input_size"],
        hidden_size=config["prior_hidden_size"],
        num_layers=config["prior_num_layers"],
        dropout=config["prior_dropout"],
    ).to(device)
    prior_model, _ = checkpointing_utils.load_checkpoint(config["prior_checkpoint"])
    program_prior.load_state_dict(prior_model)
    program_prior.eval()

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["qc_model_input_size"],
        hidden_size=config["qc_model_hidden_size"],
        num_layers=config["qc_model_num_layers"],
        dropout=config["qc_model_dropout"],
    ).to(device)
    __pg_model, _ = checkpointing_utils.load_checkpoint(config["pg_checkpoint"])
    program_generator.load_state_dict(__pg_model)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=config["qc_model_input_size"],
        hidden_size=config["qc_model_hidden_size"],
        num_layers=config["qc_model_num_layers"],
        dropout=config["qc_model_dropout"],
    ).to(device)
    __qr_model, _ = checkpointing_utils.load_checkpoint(config["qr_checkpoint"])
    question_reconstructor.load_state_dict(__qr_model)

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(config["mt_model_image_feature_size"]),
        module_channels=config["mt_model_module_channels"],
        class_projection_channels=config["mt_model_class_projection_channels"],
        classifier_linear_size=config["mt_model_classifier_linear_size"]
    ).to(device)
    __nmn_model, _ = checkpointing_utils.load_checkpoint(config["nmn_checkpoint"])
    nmn.load_state_dict(__nmn_model)

    elbo = JointTrainingNegativeElbo(
        program_generator, question_reconstructor, program_prior, nmn,
        beta=config["qc_beta"], gamma=config["jt_gamma"], baseline_decay=config["qc_delta"],
        objective=config["jt_objective"]
    )
    optimizer = optim.Adam(
        itertools.chain(
            program_generator.parameters(), question_reconstructor.parameters(), nmn.parameters()
        ),
        lr=config["optim_lr_initial"], weight_decay=config["optim_weight_decay"]
    )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config["optim_lr_gamma"],
        patience=config["optim_lr_patience"], threshold=1e-2
    )

    if -1 not in args.gpu_ids:
        # Don't wrap to DataParallel for CPU-mode.
        nmn = nn.DataParallel(nmn, args.gpu_ids)

    # ============================================================================================
    #   BEFORE TRAINING LOOP
    # ============================================================================================
    __pg_checkpoint_manager = checkpointing_utils.CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=program_generator,
        optimizer=optimizer,
        filename_prefix="program_generator",
    )
    __qr_checkpoint_manager = checkpointing_utils.CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=question_reconstructor,
        optimizer=optimizer,
        filename_prefix="question_reconstructor",
    )
    __nmn_checkpoint_manager = checkpointing_utils.CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=nmn,
        optimizer=optimizer,
        filename_prefix="nmn",
    )
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)

    # Make train dataloader iteration cyclical.
    train_dataloader = common_utils.cycle(train_dataloader)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    print(f"Training for {config['optim_num_iterations']} iterations:")
    for iteration in tqdm(range(1, config["optim_num_iterations"] + 1)):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys: {"predictions", "loss", "answer_accuracy"}
        iteration_output_dict = do_iteration(
            config, batch, nmn, program_generator, question_reconstructor, program_prior,
            elbo, optimizer
        )

        # Log losses and hyperparameters.
        summary_writer.add_scalars("loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars("elbo", iteration_output_dict["elbo"], iteration)
        summary_writer.add_scalars("nmn_metrics", iteration_output_dict["nmn_metrics"], iteration)
        summary_writer.add_scalars("schedule", {"lr": optimizer.param_groups[0]["lr"]}, iteration)

        # ========================================================================================
        #   VALIDATION
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Validation after iteration {iteration}:")
            nmn.eval()
            program_generator.eval()
            question_reconstructor.eval()

            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)

                # Just accumulate metrics across batches, in these models, by a forward pass.
                with torch.no_grad():
                    sampled_programs = program_generator(
                        batch["question"], batch["program"])["predictions"]
                    _ = question_reconstructor(batch["program"], batch["question"])
                    _ = nmn(batch["image"], sampled_programs, batch["answer"])

                if (i + 1) * len(batch["question"]) > args.num_val_examples: break

            __pg_metrics = program_generator.get_metrics()
            __qr_metrics = question_reconstructor.get_metrics()
            if isinstance(nmn, nn.DataParallel):
                __nmn_metrics = nmn.module.get_metrics()
            else:
                __nmn_metrics = nmn.get_metrics()

            # Log three metrics to tensorboard.
            # keys: {"BLEU", "perplexity", "sequence_accuracy"}
            for metric_name in __pg_metrics:
                summary_writer.add_scalars(
                    "metrics/" + metric_name, {
                        "program_generation": __pg_metrics[metric_name],
                        "question_reconstruction": __qr_metrics[metric_name]
                    },
                    iteration
                )
            summary_writer.add_scalar(
                "metrics/nmn_answer_accuracy", __nmn_metrics["answer_accuracy"], iteration
            )
            lr_scheduler.step(__nmn_metrics["answer_accuracy"])

            __pg_checkpoint_manager.step(__pg_metrics["sequence_accuracy"], iteration)
            __qr_checkpoint_manager.step(__qr_metrics["sequence_accuracy"], iteration)
            __nmn_checkpoint_manager.step(__nmn_metrics["answer_accuracy"], iteration)

            nmn.train()
            program_generator.train()
            question_reconstructor.train()
