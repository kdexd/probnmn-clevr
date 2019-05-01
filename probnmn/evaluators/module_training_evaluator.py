import logging
from typing import Any, Dict, List, Type

from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ModuleTrainingDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class ModuleTrainingEvaluator(_Evaluator):
    r"""
    Performs evaluation for ``module_training`` phase, using batches of evaluation examples from
    :class:`~probnmn.data.datasets.ModuleTrainingDataset`.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other for evaluation. This should come from
        :class:`~probnmn.trainers.module_training_trainer.ModuleTrainingTrainer`.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.
    cpu_workers: int, optional (default = 0)
        Number of CPU workers to use for fetching batch examples in dataloader.

    Examples
    --------
    To evaluate a pre-trained checkpoint:

    >>> config = Config("config.yaml")  # PHASE must be "module_training"
    >>> trainer = ModuleTrainingTrainer(config, serialization_dir="/tmp")
    >>> trainer.load_checkpoint("/path/to/module_training_checkpoint.pth")
    >>> evaluator = ModuleTrainingEvaluator(config, trainer.models)
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

        if self._C.PHASE != "module_training":
            raise ValueError(
                f"Trying to initialize a ModuleTrainingEvaluator, expected config PHASE to be "
                f"module_training, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = ModuleTrainingDataset(
            self._C.DATA.VAL_TOKENS, self._C.DATA.VAL_FEATURES, in_memory=False
        )
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, num_workers=cpu_workers
        )

        super().__init__(config=config, dataloader=dataloader, models=models, gpu_ids=gpu_ids)

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._nmn = self._models["nmn"]

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Perform one iteration, given a batch. Take a forward pass to accumulate metrics in
        :class:`~probnmn.models.nmn.NeuralModulenetwork`.

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of evaluation examples sampled from dataloader.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing model predictions and/or batch validation losses of
            :class:`~probnmn.models.program_generator.ProgramGenerator` and
            :class:`~probnmn.models.nmn.NeuralModuleNetwork`. Nested dict structure::

                {
                    "program_generator": {"predictions"}
                    "nmn": {"predictions", "loss"}
                }
        """

        pg_output_dict = self._program_generator(
            batch["question"], batch["program"], decoding_strategy="greedy"
        )
        nmn_output_dict = self._nmn(batch["image"], pg_output_dict["predictions"], batch["answer"])

        return {"program_generator": pg_output_dict, "nmn": nmn_output_dict}
