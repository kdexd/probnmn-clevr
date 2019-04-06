import logging
from typing import Any, Dict, List, Type

from allennlp.data import Vocabulary
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import JointTrainingDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class JointTrainingEvaluator(_Evaluator):
    r"""
    Performs evaluation for ``joint_training`` phase, using batches of evaluation examples from
    :class:`~probnmn.data.datasets.JointTrainingDataset`.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other for evaluation. This should come from
        :class:`~probnmn.trainers.joint_training_trainer.JointTrainingTrainer`.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.

    Examples
    --------
    To evaluate a pre-trained checkpoint:

    >>> config = Config("config.yaml")  # PHASE must be "joint_training"
    >>> trainer = JointTrainingTrainer(config, serialization_dir="/tmp")
    >>> trainer.load_checkpoint("/path/to/joint_training_checkpoint.pth")
    >>> evaluator = JointTrainingEvaluator(config, trainer.models)
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

        if self._C.PHASE != "joint_training":
            raise ValueError(
                f"Trying to initialize a JointTrainingEvaluator, expected config PHASE to be "
                f"joint_training, found {self._C.PHASE}"
            )

        # Initialize vocabulary, dataloader and model.
        self._vocabulary = Vocabulary.from_files(self._C.DATA.VOCABULARY)

        # There is no notion of "supervision" during evaluation.
        dataset = JointTrainingDataset(self._C.DATA.VAL_TOKENS, self._C.DATA.VAL_FEATURES)
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
        :class:`~probnmn.models.program_generator.ProgramGenerator` and
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
                    "program_generator": {"predictions", "loss"}
                    "nmn": {"predictions", "loss"}
                }
        """

        pg_output_dict = self._program_generator(batch["question"], batch["program"])
        nmn_output_dict = self._nmn(batch["image"], pg_output_dict["predictions"], batch["answer"])

        return {"program_generator": pg_output_dict, "nmn": nmn_output_dict}
