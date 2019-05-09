import logging
from typing import Any, Dict, List, Optional

from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ProgramPriorDataset
from probnmn.models import ProgramPrior
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class ProgramPriorTrainer(_Trainer):
    r"""
    Performs training for ``program_prior`` phase, using batches of training examples from
    :class:`~probnmn.data.datasets.ProgramPriorDataset`.

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
    >>> config = Config("config.yaml")  # PHASE must be "program_prior"
    >>> trainer = ProgramPriorTrainer(config, serialization_dir="/tmp")
    >>> evaluator = ProgramPriorEvaluator(config, trainer.models)
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

        if self._C.PHASE != "program_prior":
            raise ValueError(
                f"Trying to initialize a ProgramPriorTrainer, expected config PHASE to be "
                f"program_prior, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = ProgramPriorDataset(self._C.DATA.TRAIN_TOKENS)
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, shuffle=True, num_workers=cpu_workers
        )

        # This will be a part of `self._models`, keep this handle for convenience.
        program_prior = ProgramPrior.from_config(self._C)

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={"program_prior": program_prior},
            serialization_dir=serialization_dir,
            gpu_ids=gpu_ids,
        )

        # This will be a part of `self._models`, keep this handle for convenience.
        self._program_prior = self._models["program_prior"]

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # keys: {"predictions", "loss"}
        iteration_output_dict = self._program_prior(batch["program"])
        batch_loss = iteration_output_dict["loss"].mean()
        batch_loss.backward()

        # Clamp all gradients between (-5, 5).
        for parameter in self._program_prior.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)

        return {"loss": batch_loss}

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):
        r"""
        Set ``"metric"`` key in ``val_metrics``, this governs learning rate scheduling and keeping
        track of best checkpoint (in ``super`` method). This metric will be perplexity of
        :class:`~probnmn.models.program_prior.ProgramPrior` (lower is better).

        Super method will perform learning rate scheduling, serialize checkpoint, and log all
        the validation metrics to tensorboard.

        Parameters
        ----------
        val_metrics: Dict[str, Any]
            Validation metrics of :class:`~probnmn.models.program_prior.ProgramPrior`.
            Returned by ``evaluate`` method of
            :class:`~probnmn.evaluators.program_prior_evaluator.ProgramPriorEvaluator`.
        iteration: int, optional (default = None)
            Iteration number. If ``None``, use the internal :attr:`self._iteration` counter.
        """
        # Reciprocate perplexity to make it "higher is better".
        val_metrics["metric"] = 1.0 / val_metrics["program_prior"]["perplexity"]
        super().after_validation(val_metrics, iteration)
