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
from probnmn.data.sampler import QuestionCurriculumSampler
from probnmn.models import ProgramGenerator
from probnmn.models.nmn import NeuralModuleNetwork
import probnmn.utils.checkpointing as checkpointing_utils
import probnmn.utils.common as common_utils


parser = argparse.ArgumentParser("Jointly train Program Generator and Neural Module Network on"
                                 "CLEVR v1.0 using warm starts from question coding/module training.")
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


def do_iteration(iteration: int,
                 config: Dict[str, Union[int, float, str]],
                 batch: Dict[str, torch.Tensor],
                 program_prior: ProgramPrior,
                 program_generator: ProgramGenerator,
                 question_reconstructor: QuestionReconstructor,
                 nmn: NeuralModuleNetwork,
                 moving_average_baseline: torch.Tensor,
                 optimizer: Optional[optim.Optimizer] = None):
    """Perform one iteration - forward, backward passes (and optim step, if training)."""

    # Separate out examples with supervision and without supervision, these two maskes will be
    # mutually exclusive. Shape: (batch_size, 1)
    gt_supervision_mask = batch["supervision"]
    no_gt_supervision_mask = 1 - gt_supervision_mask

    # --------------------------------------------------------------------------------------------
    # Supervision loss: \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} ) 
    # keys: {"predictions", "loss", "sequence_logprobs"}
    __pg_output_dict_gt_supervision = program_generator(batch["question"], batch["program"])
    __qr_output_dict_gt_supervision = question_reconstructor(batch["program"], batch["question"])
    # Zero out loss contribution from examples having no program supervision.
    program_generation_loss_gt_supervision = allennlp_util.masked_mean(
        __pg_output_dict_gt_supervision["loss"], gt_supervision_mask, dim=-1
    )
    question_reconstruction_loss_gt_supervision = allennlp_util.masked_mean(
        __qr_output_dict_gt_supervision["loss"], gt_supervision_mask, dim=-1
    )
    # --------------------------------------------------------------------------------------------

    # keys: {"predictions", "loss", "sequence_logprobs"}
    __pg_output_dict_no_gt_supervision = program_generator(batch["question"])
    sampled_programs = __pg_output_dict_no_gt_supervision["predictions"]

    # For examples with supervision, replace sampled programs with ground truth programs.
    gt_supervision_indices = gt_supervision_mask.nonzero().squeeze()
    sampled_programs[gt_supervision_indices] = batch["program"][gt_supervision_indices]

    nmn_output_dict = nmn(batch["image"], sampled_programs, batch["answer"])

    # Notations in comments: x', z' are questions and programs observed in dataset.
    # z are sampled programs given x' from the dataset.
    # In other words x' = batch["questions"] and z' = batch["programs"]
    if program_generator.training and question_reconstructor.training and nmn.training:
        optimizer.zero_grad()

        if config["qc_objective"] == "baseline":
            loss_objective = program_generation_loss_gt_supervision + \
                             question_reconstruction_loss_gt_supervision

            # Zero value variables for returning them to log on tensorboard.
            question_reconstruction_loss_no_gt_supervision = 0
            full_monte_carlo_elbo = 0
            reward = 0
            moving_average_baseline = 0
        elif config["qc_objective"] == "ours":
            # ------------------------------------------------------------------------------------
            # Full monte carlo gradient estimator
            # \log{p_\theta (x'|z)} - \beta * KL (\log{q_\phi(z|x')} || \log{p(z)})

            # Everything here will be calculated for examples with no (GT) supervision, examples
            # with GT supervision only need to maximize evidence.

            # Sample programs, for the questions of examples with no GT (program) supervision.
            # Sample z ~ q_\phi(z|x'), shape: (batch_size, max_program_length)

            # keys: {"predictions", "loss", "sequence_logprobs"}
            __pg_output_dict_no_gt_supervision = program_generator(batch["question"])
            sampled_programs = __pg_output_dict_no_gt_supervision["predictions"]

            # keys: {"predictions", "loss", "sequence_logprobs"}
            __qr_output_dict_no_gt_supervision = question_reconstructor(
                sampled_programs, batch["question"]
            )
            question_reconstruction_loss_no_gt_supervision = allennlp_util.masked_mean(
                __qr_output_dict_no_gt_supervision["loss"],
                no_gt_supervision_mask, dim=-1
            )

            # Mask all of these while computing net REINFORCE loss.
            # shape: (batch_size, )
            logprobs_reconstruction = __qr_output_dict_no_gt_supervision["sequence_logprobs"]
            logprobs_generation = __pg_output_dict_no_gt_supervision["sequence_logprobs"]
            logprobs_prior = program_prior(sampled_programs)["sequence_logprobs"]

            # REINFORCE reward (R): ( \log{ (p_\theta (x'|z) * p(z) ^ \beta) / (q_\phi (z|x') ) })
            # shape: (batch_size, )
            reinforce_reward = logprobs_reconstruction + \
                               config["qc_beta"] * (logprobs_prior - logprobs_generation)

            # Detach the reward term, we don't want gradients to flow to through that and get
            # counted twice, once already counted through path derivative loss.
            reinforce_reward = reinforce_reward.detach()
            # Zero out the net reward for examples with GT (program) supervision.
            centered_reward = (reinforce_reward - moving_average_baseline) * \
                              no_gt_supervision_mask.float()

            # Set reward for examples with GT (program) supervision to zero.
            reinforce_loss = - allennlp_util.masked_mean(
                centered_reward * logprobs_generation,
                no_gt_supervision_mask,
                dim=-1
            )
            path_derivative_generation_loss = - allennlp_util.masked_mean(
                logprobs_generation, no_gt_supervision_mask, dim=-1
            )
            full_monte_carlo_elbo = question_reconstruction_loss_no_gt_supervision + \
                                  config["qc_beta"] * path_derivative_generation_loss + \
                                  reinforce_loss

            # B := B + (1 - \delta * (R - B))
            moving_average_baseline += config["qc_delta"] * allennlp_util.masked_mean(
                centered_reward, no_gt_supervision_mask, dim=-1
            )

            full_elbo = allennlp_util.masked_mean(reinforce_reward, no_gt_supervision_mask, dim=-1)
            # ------------------------------------------------------------------------------------

            loss_objective = config["qc_alpha"] * question_reconstruction_loss_gt_supervision + \
                             config["qc_alpha"] * program_generation_loss_gt_supervision + \
                             full_monte_carlo_elbo

        nmn_loss = nmn_output_dict["loss"].mean()
        loss_objective = loss_objective + config["jt_gamma"] * nmn_loss

        loss_objective.backward()
        # Clamp all gradients between (-5, 5)
        for parameter in program_generator.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        for parameter in question_reconstructor.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        optimizer.step()

    iteration_output_dict = {
        "predictions": {
            # These only matter during validation.
            "__pg": __pg_output_dict_gt_supervision["predictions"],
            "__qr": __qr_output_dict_gt_supervision["predictions"],
            "nmn": nmn_output_dict["predictions"]
        },
    }
    if program_generator.training and question_reconstructor.training and nmn.training:
        iteration_output_dict = {
            "loss": {
                "question_reconstruction_no_gt": question_reconstruction_loss_no_gt_supervision,
                "question_reconstruction_gt": question_reconstruction_loss_gt_supervision,
                "full_monte_carlo_elbo": full_monte_carlo_elbo,
                "program_generation_gt": program_generation_loss_gt_supervision,
                "nmn": nmn_loss,
                "objective": loss_objective
            },
            "elbo": {
                "reconstruction": -allennlp.masked_mean(
                    logprobs_reconstruction, no_gt_supervision_mask, dim=-1
                )
                "kl_divergence": -allennlp.masked_mean(
                    logprobs_prior - logprobs_generation, no_gt_supervision_mask, dim=-1
                )
                "full_elbo": full_elbo
            }
            "metrics": {
                "answer_accuracy": nmn_output_dict["answer_accuracy"],
                "average_invalid": nmn_output_dict["average_invalid"]
            }
        }
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
        args.tokens_train_h5, args.features_train_h5
    )
    val_dataset = JointTrainingDataset(
        args.tokens_val_h5, args.features_val_h5
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
        average_loss_across_timesteps=config["qc_average_loss_across_timesteps"],
        average_logprobs_across_timesteps=config["qc_average_logprobs_across_timesteps"],
    ).to(device)
    prior_model, prior_optimizer = checkpointing_utils.load_checkpoint(config["prior_checkpoint"])
    program_prior.load_state_dict(prior_model)
    program_prior.eval()

    program_generator = ProgramGenerator(
        vocabulary=vocabulary,
        input_size=config["qc_model_input_size"],
        hidden_size=config["qc_model_hidden_size"],
        num_layers=config["qc_model_num_layers"],
        dropout=config["qc_model_dropout"],
        average_loss_across_timesteps=config["qc_average_loss_across_timesteps"],
        average_logprobs_across_timesteps=config["qc_average_logprobs_across_timesteps"],
    ).to(device)

    question_reconstructor = QuestionReconstructor(
        vocabulary=vocabulary,
        input_size=config["qc_model_input_size"],
        hidden_size=config["qc_model_hidden_size"],
        num_layers=config["qc_model_num_layers"],
        dropout=config["qc_model_dropout"],
        average_loss_across_timesteps=config["qc_average_loss_across_timesteps"],
        average_logprobs_across_timesteps=config["qc_average_logprobs_across_timesteps"],
    ).to(device)

    nmn = NeuralModuleNetwork(
        vocabulary=vocabulary,
        image_feature_size=tuple(config["mt_model_image_feature_size"]),
        module_channels=config["mt_model_module_channels"],
        class_projection_channels=config["mt_model_class_projection_channels"],
        classifier_linear_size=config["mt_model_classifier_linear_size"]
    ).to(device)

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
    program_generator_checkpoint_manager = checkpointing_utils.CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=program_generator,
        optimizer=optimizer,
        filename_prefix="program_generator",
        mode="min",
    )
    question_reconstructor_checkpoint_manager = checkpointing_utils.CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=question_reconstructor,
        optimizer=optimizer,
        filename_prefix="question_reconstructor",
        mode="min",
    )
    nmn_checkpoint_manager = checkpointing_utils.CheckpointManager(
        checkpoint_dirpath=args.save_dirpath,
        config=config,
        model=nmn,
        optimizer=optimizer,
        mode="max",
        filename_prefix="nmn",
    )
    summary_writer = SummaryWriter(log_dir=args.save_dirpath)

    # Make train dataloader iteration cyclical.
    train_dataloader = common_utils.cycle(train_dataloader)

    moving_average_baseline = torch.tensor(0.0)

    # ============================================================================================
    #   TRAINING LOOP
    # ============================================================================================
    print(f"Training for {config['optim_num_iterations']} iterations:")
    for iteration in tqdm(range(start_iteration, config["optim_num_iterations"])):
        batch = next(train_dataloader)
        for key in batch:
            batch[key] = batch[key].to(device)

        # keys: {"predictions", "loss", "answer_accuracy"}
        iteration_output_dict = do_iteration(
            iteration, config, batch, program_prior, program_generator, question_reconstructor,
            nmn, moving_average_baseline, optimizer
        )

        # Log losses and hyperparameters.
        summary_writer.add_scalars("loss", iteration_output_dict["loss"], iteration)
        summary_writer.add_scalars("elbo": iteration_output_dict["elbo"], iteration)
        summary_writer.add_scalars("schedule", {"lr": optimizer.param_groups[0]["lr"]}, iteration)
        summary_writer.add_scalars("train_metrics", iteration_output_dict["metrics"], iteration)

        # ========================================================================================
        #   VALIDATE AND (TODO) PRINT FEW EXAMPLES
        # ========================================================================================
        if iteration % args.checkpoint_every == 0:
            print(f"Validation after iteration {iteration}:")
            nmn.eval()
            program_generator.eval()
            question_reconstructor.eval()

            for i, batch in enumerate(tqdm(val_dataloader)):
                for key in batch:
                    batch[key] = batch[key].to(device)

                with torch.no_grad():
                    iteration_output_dict = do_iteration(
                        iteration, config, batch, program_prior, program_generator,
                        question_reconstructor, nmn, moving_average_baseline, optimizer
                    )
                if (i + 1) * len(batch["question"]) > args.num_val_examples: break

            if isinstance(nmn, nn.DataParallel):
                nmn_metrics = nmn.module.get_metrics()
            else:
                nmn_metrics = nmn.get_metrics()

            __pg_metrics = program_generator.get_metrics()
            __qr_metrics = question_reconstructor.get_metrics()

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

            answer_accuracy = nmn_metrics["answer_accuracy"]
            average_invalid = nmn_metrics["average_invalid"]

            lr_scheduler.step(nmn_metrics["answer_accuracy"])

            summary_writer.add_scalar("val/answer_accuracy", answer_accuracy, iteration)
            summary_writer.add_scalar("val/average_invalid", average_invalid, iteration)
            program_generator_checkpoint_manager.step(__pg_metrics["perplexity"], iteration)
            question_reconstructor_checkpoint_manager.step(__qr_metrics["perplexity"], iteration)
            nmn_checkpoint_manager.step(answer_accuracy, iteration)
            nmn.train()
            program_generator.train()
            question_reconstructor.train()

    # ============================================================================================
    #   AFTER TRAINING END
    # ============================================================================================
    program_generator_checkpoint_manager.save_best()
    question_reconstructor_checkpoint_manager.save_best()
    nmn_checkpoint_manager.save_best()
    summary_writer.close()
