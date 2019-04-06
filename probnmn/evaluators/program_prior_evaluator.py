import logging
from typing import Any, Dict, List, Optional, Type

from allennlp.data import Vocabulary
import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ProgramPriorDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class ProgramPriorEvaluator(_Evaluator):
    r"""
    Performs evaluation for ``program_prior`` phase, using batches of evaluation examples from
    :class:`~probnmn.data.datasets.ProgramPriorDataset`.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other for evaluation. This should come from
        :class:`~probnmn.trainers.program_prior_trainer.ProgramPriorTrainer`.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.

    Examples
    --------
    To evaluate a pre-trained checkpoint:

    >>> config = Config("config.yaml")  # PHASE must be "program_prior"
    >>> trainer = ProgramPriorTrainer(config, serialization_dir="/tmp")
    >>> trainer.load_checkpoint("/path/to/program_prior_checkpoint.pth")
    >>> evaluator = ProgramPriorEvaluator(config, trainer.models)
    >>> eval_metrics = evaluator.evaluate(num_batches=50)
    """

    def __init__(
        self,
        config: Config,
        models: Dict[str, Type[nn.Module]],
        gpu_ids: List[int] = [0],
        cpu_workers: int = 0,
    ):
        self._C = config

        if self._C.PHASE != "program_prior":
            raise ValueError(
                f"Trying to initialize a ProgramPriorEvaluator, expected config PHASE to be "
                f"program_prior, found {self._C.PHASE}"
            )

        # Initialize vocabulary, dataloader and model.
        self._vocabulary = Vocabulary.from_files(self._C.DATA.VOCABULARY)

        dataset = ProgramPriorDataset(self._C.DATA.VAL_TOKENS)
        dataloader = DataLoader(dataset, batch_size=self._C.OPTIM.BATCH_SIZE)

        super().__init__(config=config, dataloader=dataloader, models=models, gpu_ids=gpu_ids)

        # This will be a part of `self._models`, keep this handle for convenience.
        self._program_prior = self._models["program_prior"]

    def evaluate(self, num_batches: Optional[int] = None):
        r"""
        Perform evaluation using first ``num_batches`` of dataloader and return all evaluation
        metrics from the models. After evaluation, also print some qualitative examples.

        Parameters
        ----------
        num_batches: int, optional (default=None)
            Number of batches to use from dataloader. If ``None``, use all batches.

        Returns
        -------
        Dict[str, Any]
            Final evaluation metrics for all the models. For ``program_prior`` phase, this dict
            will have keys: ``{"perplexity"}``.
        """

        eval_metrics = super().evaluate(num_batches)

        # ----------------------------------------------------------------------------------------
        # PRINT MODEL PREDICTIONS FOR FIVE EXAMPLES (OF FIRST BATCH)
        # ----------------------------------------------------------------------------------------
        self._program_prior.eval()
        for batch in self._dataloader:
            for key in batch:
                batch[key] = batch[key].to(self._device)
            break

        with torch.no_grad():
            output_dict = self._do_iteration(batch)["program_prior"]

        print("\n")
        for inp, out in zip(batch["program"][:5], output_dict["predictions"][:5]):
            # Print only first five time-steps, these sequences can be really long.
            input_program = " ".join(
                self._vocabulary.get_token_from_index(i.item(), "programs") for i in inp[:6]
            )
            output_program = " ".join(
                self._vocabulary.get_token_from_index(o.item(), "programs") for o in out[:6]
            )
            logger.info(f"INPUT PROGRAM: {input_program} ...")
            logger.info(f"OUTPUT PROGRAM: {output_program} ...")
            logger.info("-" * 60)

        self._program_prior.train()
        # ----------------------------------------------------------------------------------------

        return eval_metrics

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Perform one iteration, given a batch. Take a forward pass to accumulate metrics in
        :class:`~probnmn.models.program_prior.ProgramPrior`.

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of evaluation examples sampled from dataloader.

        Returns
        -------
        Dict[str, Any]
            An output dictionary containing predictions of next-time step, and loss (by teacher
            forcing). Nested dict structure::

                {
                    "program_prior": {"predictions", "loss"}
                }

        """

        return {"program_prior": self._program_prior(batch["program"])}
