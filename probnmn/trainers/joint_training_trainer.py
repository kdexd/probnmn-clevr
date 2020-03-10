import itertools
import logging
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import JointTrainingDataset
from probnmn.data.samplers import SupervisionWeightedRandomSampler
from probnmn.models import (
    ProgramPrior,
    ProgramGenerator,
    QuestionReconstructor,
    NeuralModuleNetwork,
)
from probnmn.modules.elbo import JointTrainingElbo
from probnmn.utils.checkpointing import CheckpointManager
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class JointTrainingTrainer(_Trainer):
    r"""
    Performs training for ``joint_training`` phase, using batches of training examples from
    :class:`~probnmn.data.datasets.JointTrainingDataset`.

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
    >>> config = Config("config.yaml")  # PHASE must be "joint_training"
    >>> trainer = JointTrainingTrainer(config, serialization_dir="/tmp")
    >>> evaluator = JointTrainingEvaluator(config, trainer.models)
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

        if self._C.PHASE != "joint_training":
            raise ValueError(
                f"Trying to initialize a JointTrainingTrainer, expected config PHASE to be "
                f"joint_training, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = JointTrainingDataset(
            self._C.DATA.TRAIN_TOKENS,
            self._C.DATA.TRAIN_FEATURES,
            num_supervision=self._C.SUPERVISION,
            supervision_question_max_length=self._C.SUPERVISION_QUESTION_MAX_LENGTH,
        )
        sampler = SupervisionWeightedRandomSampler(dataset)
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, sampler=sampler, num_workers=cpu_workers
        )

        program_generator = ProgramGenerator.from_config(self._C)
        question_reconstructor = QuestionReconstructor.from_config(self._C)
        nmn = NeuralModuleNetwork.from_config(self._C)

        # Load checkpoints from question_coding and module_training phases.
        CheckpointManager(
            program_generator=program_generator, question_reconstructor=question_reconstructor
        ).load(self._C.CHECKPOINTS.QUESTION_CODING)

        CheckpointManager(nmn=nmn).load(self._C.CHECKPOINTS.MODULE_TRAINING)

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={
                "program_generator": program_generator,
                "question_reconstructor": question_reconstructor,
                "nmn": nmn,
            },
            serialization_dir=serialization_dir,
            gpu_ids=gpu_ids,
        )

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._question_reconstructor = self._models["question_reconstructor"]
        self._nmn = self._models["nmn"]

        # Load program prior from checkpoint, this will be frozen during question coding.
        self._program_prior = ProgramPrior.from_config(self._C).to(self._device)
        CheckpointManager(program_prior=self._program_prior).load(
            self._C.CHECKPOINTS.PROGRAM_PRIOR
        )
        self._program_prior.eval()

        # Instantiate an elbo module to compute evidence lower bound during `_do_iteration`.
        self._elbo = JointTrainingElbo(
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
        # Separate out examples with supervision and without supervision, these two lists will be
        # mutually exclusive.
        supervision_indices = batch["supervision"].nonzero().squeeze()
        no_supervision_indices = (1 - batch["supervision"]).nonzero().squeeze()

        # Pick a subset of questions without (GT) program supervision, sample programs and pass
        # through the neural module network.
        question_tokens_no_supervision = batch["question"][no_supervision_indices]
        image_features_no_supervision = batch["image"][no_supervision_indices]
        answer_tokens_no_supervision = batch["answer"][no_supervision_indices]

        # keys: {"reconstruction_likelihood", "kl_divergence", "elbo", "reinforce_reward",
        #        "nmn_loss"}
        elbo_output_dict = self._elbo(
            question_tokens_no_supervision,
            image_features_no_supervision,
            answer_tokens_no_supervision,
        )
        nmn_loss = elbo_output_dict.pop("nmn_loss")
        loss_objective = self._C.GAMMA * nmn_loss - elbo_output_dict["elbo"]

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
                question_tokens_supervision,
                program_tokens_supervision,
                decoding_strategy="sampling",
            )
            qr_output_dict_supervision = self._question_reconstructor(
                program_tokens_supervision,
                question_tokens_supervision,
                decoding_strategy="sampling",
            )
            program_generation_loss_supervision = pg_output_dict_supervision["loss"].mean()
            question_reconstruction_loss_supervision = qr_output_dict_supervision["loss"].mean()
            # ------------------------------------------------------------------------------------

            loss_objective = loss_objective + self._C.ALPHA * (
                program_generation_loss_supervision + question_reconstruction_loss_supervision
            )

        loss_objective.backward()

        # Clamp all gradients between (-5, 5)
        for parameter in itertools.chain(
            self._program_generator.parameters(),
            self._question_reconstructor.parameters(),
            self._nmn.parameters(),
        ):
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)

        iteration_output_dict = {"loss": {"nmn": nmn_loss}, "elbo": elbo_output_dict}
        if self._C.OBJECTIVE == "ours":
            iteration_output_dict["loss"].update(
                {
                    "question_reconstruction_gt": question_reconstruction_loss_supervision,
                    "program_generation_gt": program_generation_loss_supervision,
                }
            )
        return iteration_output_dict

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):
        r"""
        Set ``"metric"`` key in ``val_metrics``, this governs learning rate scheduling and keeping
        track of best checkpoint (in ``super`` method). This metric will be answer accuracy.

        Super method will perform learning rate scheduling, serialize checkpoint, and log all
        the validation metrics to tensorboard.

        Parameters
        ----------
        val_metrics: Dict[str, Any]
            Validation metrics of :class:`~probnmn.models.nmn.NeuralModuleNetwork`.
            Returned by ``evaluate`` method of
            :class:`~probnmn.evaluators.joint_training_evaluator.JointTrainingEvaluator`.
        iteration: int, optional (default = None)
            Iteration number. If ``None``, use the internal :attr:`self._iteration` counter.
        """
        val_metrics["metric"] = val_metrics["nmn"]["answer_accuracy"]
        super().after_validation(val_metrics, iteration)
