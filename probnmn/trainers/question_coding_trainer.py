import itertools
import logging
from typing import Any, Dict, List, Optional

from allennlp.data import Vocabulary
import torch
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import QuestionCodingDataset
from probnmn.data.samplers import SupervisionWeightedRandomSampler
from probnmn.models import ProgramPrior, ProgramGenerator, QuestionReconstructor
from probnmn.modules.elbo import QuestionCodingElbo
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class QuestionCodingTrainer(_Trainer):
    r"""
    Performs training for ``question_coding`` phase, using batches of training examples from
    :class:`~probnmn.data.datasets.QuestionCodingDataset`.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    serialization_dir: str
        Path to a directory for tensorboard logging and serializing checkpoints.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.
    cpu_workers: int, optional (default = 0)
        Number of CPU workers to use for fetching batch examples in dataloader.

    Examples
    --------
    >>> config = Config("config.yaml")  # PHASE must be "question_coding"
    >>> trainer = QuestionCodingTrainer(config, serialization_dir="/tmp")
    >>> evaluator = QuestionCodingEvaluator(config, trainer.models)
    >>> for iteration in range(100):
    >>>     trainer.step()
    >>>     # validation every 100 steps
    >>>     if iteration % 100 == 0:
    >>>         val_metrics = evaluator.evaluate()
    >>>         trainer.after_validation(val_metrics, iteration)
    """

    def __init__(
        self,
        config: Config,
        serialization_dir: str,
        gpu_ids: List[int] = [0],
        cpu_workers: int = 0,
    ):
        self._C = config

        if self._C.PHASE != "question_coding":
            raise ValueError(
                f"Trying to initialize a QuestionCodingTrainer, expected config PHASE to be "
                f"question_coding, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = QuestionCodingDataset(
            self._C.DATA.TRAIN_TOKENS,
            num_supervision=self._C.SUPERVISION,
            supervision_question_max_length=self._C.SUPERVISION_QUESTION_MAX_LENGTH,
        )
        sampler = SupervisionWeightedRandomSampler(dataset)
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, sampler=sampler, num_workers=cpu_workers
        )

        program_prior = ProgramPrior.from_config(self._C)
        program_generator = ProgramGenerator.from_config(self._C)
        question_reconstructor = QuestionReconstructor.from_config(self._C)

        # Load program prior from checkpoint, this will be frozen during question coding.
        program_prior.load_state_dict(
            torch.load(self._C.CHECKPOINTS.PROGRAM_PRIOR)["program_prior"]
        )
        program_prior.eval()

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={
                "program_generator": program_generator,
                "question_reconstructor": question_reconstructor,
                "program_prior": program_prior,
            },
            serialization_dir=serialization_dir,
            gpu_ids=gpu_ids,
        )

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._question_reconstructor = self._models["question_reconstructor"]
        self._program_prior = self._models["program_prior"]

        # Instantiate an elbo module to compute evidence lower bound during `_do_iteration`.
        self._elbo = QuestionCodingElbo(
            program_generator=self._program_generator,
            question_reconstructor=self._question_reconstructor,
            program_prior=self._program_prior,
            beta=self._C.BETA,
            baseline_decay=self._C.DELTA,
        )

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
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
            question_tokens_supervision, program_tokens_supervision, decoding_strategy="sampling"
        )
        qr_output_dict_supervision = self._question_reconstructor(
            program_tokens_supervision, question_tokens_supervision, decoding_strategy="sampling"
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
        for parameter in itertools.chain(
            self._program_generator.parameters(), self._question_reconstructor.parameters()
        ):
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
        val_metrics["metric"] = val_metrics["program_generator"]["sequence_accuracy"]
        super().after_validation(val_metrics, iteration)
