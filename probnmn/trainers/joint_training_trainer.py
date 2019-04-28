import logging
from typing import Any, Dict, Optional

from allennlp.data import Vocabulary
import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import JointTrainingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import (
    ProgramPrior,
    ProgramGenerator,
    QuestionReconstructor,
    NeuralModuleNetwork,
)
from probnmn.modules.elbo import JointTrainingNegativeElbo
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class JointTrainingTrainer(_Trainer):
    def __init__(self, config: Config, device: torch.device, serialization_dir: str):
        self._C = config

        if self._C.PHASE != "joint_training":
            raise ValueError(
                f"Trying to initialize a JointTrainingTrainer, expected config PHASE to be "
                f"joint_training, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = JointTrainingDataset(
            self._C.DATA.TRAIN.TOKENS,
            self._C.DATA.TRAIN.IMAGE_FEATURES,
            num_supervision=self._C.SUPERVISION,
            supervision_question_max_length=self._C.SUPERVISION_QUESTION_MAX_LENGTH,
        )
        sampler = SupervisionWeightedRandomSampler(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=self._C.OPTIM.BATCH_SIZE,
            sampler=sampler,
            # num_workers=self._A.cpu_workers,
        )

        # Vocabulary is needed to instantiate the models.
        vocabulary = Vocabulary.from_files(self._C.DATA.VOCABULARY)

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = ProgramGenerator(
            vocabulary=vocabulary,
            input_size=self._C.PROGRAM_GENERATOR.INPUT_SIZE,
            hidden_size=self._C.PROGRAM_GENERATOR.HIDDEN_SIZE,
            num_layers=self._C.PROGRAM_GENERATOR.NUM_LAYERS,
            dropout=self._C.PROGRAM_GENERATOR.DROPOUT,
        ).to(device)

        self._question_reconstructor = QuestionReconstructor(
            vocabulary=vocabulary,
            input_size=self._C.QUESTION_RECONSTRUCTOR.INPUT_SIZE,
            hidden_size=self._C.QUESTION_RECONSTRUCTOR.HIDDEN_SIZE,
            num_layers=self._C.QUESTION_RECONSTRUCTOR.NUM_LAYERS,
            dropout=self._C.QUESTION_RECONSTRUCTOR.DROPOUT,
        ).to(device)

        self._nmn = NeuralModuleNetwork(
            vocabulary=vocabulary,
            image_feature_size=tuple(self._C.NMN.IMAGE_FEATURE_SIZE),
            module_channels=self._C.NMN.MODULE_CHANNELS,
            class_projection_channels=self._C.NMN.CLASS_PROJECTION_CHANNELS,
            classifier_linear_size=self._C.NMN.CLASSIFIER_LINEAR_SIZE,
        ).to(device)

        # Load checkpoints from question coding and module training phases.
        question_coding_checkpoint = torch.load(self._C.CHECKPOINTS.QUESTION_CODING)
        self._program_generator.load_state_dict(question_coding_checkpoint["program_generator"])
        self._question_reconstructor.load_state_dict(
            question_coding_checkpoint["question_reconstructor"]
        )
        self._nmn.load_state_dict(torch.load(self._C.CHECKPOINTS.NMN)["nmn"])

        # if -1 not in self._A.gpu_ids:
        #     # Don't wrap to DataParallel for CPU-mode.
        #     self._nmn = nn.DataParallel(self._nmn, self._A.gpu_ids)

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={
                "program_generator": self._program_generator,
                "question_reconstructor": self._question_reconstructor,
                "nmn": self._nmn,
            },
            device=device,
            serialization_dir=serialization_dir,
        )

        # Program Prior checkpoint, this will be frozen during question coding.
        self._program_prior = ProgramPrior(
            vocabulary=vocabulary,
            input_size=self._C.PROGRAM_PRIOR.INPUT_SIZE,
            hidden_size=self._C.PROGRAM_PRIOR.HIDDEN_SIZE,
            num_layers=self._C.PROGRAM_PRIOR.NUM_LAYERS,
            dropout=self._C.PROGRAM_PRIOR.DROPOUT,
        ).to(device)

        self._program_prior.load_state_dict(
            torch.load(self._C.CHECKPOINTS.PROGRAM_PRIOR)["program_prior"]
        )
        self._program_prior.eval()

        # Instantiate an elbo module to compute evidence lower bound during `_do_iteration`.
        self._elbo = JointTrainingNegativeElbo(
            program_generator=self._program_generator,
            question_reconstructor=self._question_reconstructor,
            nmn=self._nmn,
            program_prior=self._program_prior,
            beta=self._C.BETA,
            gamma=self._C.GAMMA,
            baseline_decay=self._C.DELTA,
            objective=self._C.OBJECTIVE,
        )

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass and compute loss. Return an output dict
        which would be passed to `after_iteration`.
        """

        # Separate out examples with supervision and without supervision, these two lists will be
        # mutually exclusive.
        supervision_indices = batch["supervision"].nonzero().squeeze()
        no_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

        # Pick a subset of questions without (GT) program supervision, sample programs and pass
        # through the neural module network.
        question_tokens_no_supervision = batch["question"][no_supervision_indices]
        image_features_no_supervision = batch["image"][no_supervision_indices]
        answer_tokens_no_supervision = batch["answer"][no_supervision_indices]

        # keys: {"nmn_loss", "elbo"}
        elbo_output_dict = self._elbo(
            question_tokens_no_supervision,
            image_features_no_supervision,
            answer_tokens_no_supervision,
        )

        loss_objective = self._C.GAMMA * elbo_output_dict["nmn_loss"] - elbo_output_dict["elbo"]

        if self._C.OBJECTIVE == "ours":
            # ------------------------------------------------------------------------------------
            # Supervision loss (program generator + question reconstructor):
            # Ignore question reconstructor for "baseline" objective, it's gradients do not
            # interfere with program generator anyway.

            # \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} )
            program_tokens_supervision = batch["program"][supervision_indices]
            question_tokens_supervision = batch["question"][supervision_indices]

            # keys: {"predictions", "loss"}
            pg_output_dict_supervision = self._program_generator(
                question_tokens_supervision, program_tokens_supervision
            )
            qr_output_dict_supervision = self._question_reconstructor(
                program_tokens_supervision, question_tokens_supervision
            )
            program_generation_loss_supervision = pg_output_dict_supervision["loss"].mean()
            question_reconstruction_loss_supervision = qr_output_dict_supervision["loss"].mean()
            # ------------------------------------------------------------------------------------

            loss_objective += self._C.ALPHA * (
                program_generation_loss_supervision + question_reconstruction_loss_supervision
            )

        loss_objective.backward()

        # Clamp all gradients between (-5, 5)
        for parameter in self._program_generator.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        for parameter in self._question_reconstructor.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        for parameter in self._nmn.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)

        iteration_output_dict = {
            "loss": {"nmn": elbo_output_dict["nmn_loss"]},
            "elbo": {"elbo": elbo_output_dict["elbo"]},
        }
        if self._C.OBJECTIVE == "ours":
            iteration_output_dict["loss"].update(
                {
                    "question_reconstruction_gt": question_reconstruction_loss_supervision,
                    "program_generation_gt": program_generation_loss_supervision,
                }
            )
        return iteration_output_dict

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):

        # Set "metric" key in `val_metrics`, this governs learning rate scheduling and keeping
        # track of best checkpoint. For joint training, it will be answer accuracy.
        val_metrics["metric"] = val_metrics["nmn"]["answer_accuracy"]

        # Super method will perform learning rate scheduling, serialize checkpoint, and log all
        # the validation metrics to tensorboard.
        super().after_validation(val_metrics, iteration)
