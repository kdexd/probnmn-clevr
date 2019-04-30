import logging
from typing import Any, Dict, List, Optional

from allennlp.data import Vocabulary
import torch
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ModuleTrainingDataset
from probnmn.models import NeuralModuleNetwork, ProgramGenerator
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class ModuleTrainingTrainer(_Trainer):
    r"""
    Performs training for ``module_training`` phase, using batches of training examples from
    :class:`~probnmn.data.datasets.ModuleTrainingDataset`.

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
    >>> config = Config("config.yaml")  # PHASE must be "module_training"
    >>> trainer = ModuleTrainingTrainer(config, serialization_dir="/tmp")
    >>> evaluator = ModuleTrainingEvaluator(config, trainer.models)
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

        if self._C.PHASE != "module_training":
            raise ValueError(
                f"Trying to initialize a ModuleTrainingTrainer, expected config PHASE to be "
                f"module_training, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = ModuleTrainingDataset(
            self._C.DATA.TRAIN_TOKENS, self._C.DATA.TRAIN_FEATURES, in_memory=False
        )
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, num_workers=cpu_workers
        )

        # Vocabulary is needed to instantiate the models.
        vocabulary = Vocabulary.from_files(self._C.DATA.VOCABULARY)

        program_generator = ProgramGenerator(
            vocabulary=vocabulary,
            input_size=self._C.PROGRAM_GENERATOR.INPUT_SIZE,
            hidden_size=self._C.PROGRAM_GENERATOR.HIDDEN_SIZE,
            num_layers=self._C.PROGRAM_GENERATOR.NUM_LAYERS,
            dropout=self._C.PROGRAM_GENERATOR.DROPOUT,
        )

        # Load program generator from checkpoint, this will be frozen during module training.
        program_generator.load_state_dict(
            torch.load(self._C.CHECKPOINTS.QUESTION_CODING)["program_generator"]
        )
        program_generator.eval()

        nmn = NeuralModuleNetwork(  # type: ignore
            vocabulary=vocabulary,
            image_feature_size=tuple(self._C.NMN.IMAGE_FEATURE_SIZE),
            module_channels=self._C.NMN.MODULE_CHANNELS,
            class_projection_channels=self._C.NMN.CLASS_PROJECTION_CHANNELS,
            classifier_linear_size=self._C.NMN.CLASSIFIER_LINEAR_SIZE,
        )

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={"program_generator": program_generator, "nmn": nmn},
            serialization_dir=serialization_dir,
            gpu_ids=gpu_ids,
        )

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._nmn = self._models["nmn"]

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        sampled_programs = self._program_generator(batch["question"])["predictions"]
        output_dict = self._nmn(batch["image"], sampled_programs, batch["answer"])
        batch_loss = output_dict["loss"].mean()
        batch_loss.backward()

        for parameter in self._nmn.parameters():
            if parameter.grad is not None:
                parameter.grad.clamp_(min=-5, max=5)

        return {"loss": batch_loss, "metrics": output_dict["metrics"]}

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
            :class:`~probnmn.evaluators.module_training_evaluator.ModuleTrainingEvaluator`.
        iteration: int, optional (default = None)
            Iteration number. If ``None``, use the internal :attr:`self._iteration` counter.
        """
        # Remove metrics of program generator, they would be dummy values as programs were
        # sampled without providing (GT) program supervision.
        val_metrics.pop("program_generator")

        val_metrics["metric"] = val_metrics["nmn"]["answer_accuracy"]
        super().after_validation(val_metrics, iteration)
