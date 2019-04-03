import logging
from typing import Any, Dict, Optional

from allennlp.data import Vocabulary
import torch
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import QuestionCodingDataset
from probnmn.data.sampler import SupervisionWeightedRandomSampler
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from probnmn.modules.elbo import QuestionCodingElbo
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class QuestionCodingTrainer(_Trainer):
    def __init__(self, config: Config, device: torch.device, serialization_dir: str):
        self._C = config

        if self._C.PHASE != "question_coding":
            raise ValueError(
                f"Trying to initialize a QuestionCodingTrainer, expected config PHASE to be "
                f"question_coding, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = QuestionCodingDataset(
            self._C.DATA.TRAIN.TOKENS,
            num_supervision=self._C.SUPERVISION,
            supervision_question_max_length=self._C.SUPERVISION_QUESTION_MAX_LENGTH,
        )
        sampler = SupervisionWeightedRandomSampler(dataset)
        dataloader = DataLoader(dataset, batch_size=self._C.OPTIM.BATCH_SIZE, sampler=sampler)

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

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={
                "program_generator": self._program_generator,
                "question_reconstructor": self._question_reconstructor,
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
        self._elbo = QuestionCodingElbo(
            program_generator=self._program_generator,
            question_reconstructor=self._question_reconstructor,
            program_prior=self._program_prior,
            beta=self._C.BETA,
            baseline_decay=self._C.DELTA,
        )

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass and compute loss. Return an output dict
        which would be passed to `after_iteration`.
        """

        # Separate out examples with supervision and without supervision, these two lists will be
        # mutually exclusive.
        supervision_indices = batch["supervision"].nonzero().squeeze()
        no_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

        # ----------------------------------------------------------------------------------------
        # Supervision loss: \alpha * ( \log{q_\phi (z'|x')} + \log{p_\theta (x'|z')} )

        # Pick a subset of questions without (GT) program supervision, and maximize it's
        # evidence (through a variational lower bound).
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
        # ----------------------------------------------------------------------------------------

        if self._C.OBJECTIVE == "baseline":
            loss_objective = (
                program_generation_loss_supervision + question_reconstruction_loss_supervision
            )
            # Empty placeholder for elbo values (not used for "baseline" objective).
            elbo_output_dict: Dict[str, Any] = {}
        elif self._C.OBJECTIVE == "ours":
            # Pick a subset of questions without (GT) program supervision, and maximize it's
            # evidence (through a variational lower bound).
            question_tokens_no_supervision = batch["question"][no_supervision_indices]

            # keys: {"reconstruction_likelihood", "kl_divergence", "elbo", "reinforce_reward"}
            elbo_output_dict = self._elbo(question_tokens_no_supervision)

            # Evidence lower bound is to be maximized, so negate it while adding to loss.
            loss_objective = -elbo_output_dict["elbo"] + self._C.ALPHA * (
                question_reconstruction_loss_supervision + program_generation_loss_supervision
            )

        loss_objective.backward()
        # Clamp all gradients between (-5, 5)
        for parameter in self._program_generator.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)
        for parameter in self._question_reconstructor.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)

        return {
            "loss": {
                "question_reconstruction_gt": question_reconstruction_loss_supervision,
                "program_generation_gt": program_generation_loss_supervision,
            },
            "elbo": elbo_output_dict,
        }

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):

        # Set "metric" key in `val_metrics`, this governs learning rate scheduling and keeping
        # track of best checkpoint. For question coding, it will be program generation accuracy.
        val_metrics["metric"] = val_metrics["program_generator"]["sequence_accuracy"]

        # Super method will perform learning rate scheduling, serialize checkpoint, and log all
        # the validation metrics to tensorboard.
        super().after_validation(val_metrics, iteration)
