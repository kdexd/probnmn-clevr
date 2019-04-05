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
    def __init__(self, config: Config, serialization_dir: str, gpu_ids: List[int] = [0]):
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
            dataset,
            batch_size=self._C.OPTIM.BATCH_SIZE,
            # num_workers=self._A.cpu_workers
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

        nmn = NeuralModuleNetwork(
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
        """Perform one iteration, take a forward pass and compute loss. Return an output dict
        which would be passed to `after_iteration`.
        """

        sampled_programs = self._program_generator(batch["question"])["predictions"]
        output_dict = self._nmn(batch["image"], sampled_programs, batch["answer"])
        batch_loss = output_dict["loss"].mean()
        batch_loss.backward()

        return {
            "loss": batch_loss,
            "metrics": {
                "answer_accuracy": output_dict["answer_accuracy"],
                "average_invalid": output_dict["average_invalid"],
            },
        }

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):

        # Set "metric" key in `val_metrics`, this governs learning rate scheduling and keeping
        # track of best checkpoint. For module training, it will be answer accuracy.
        val_metrics["metric"] = val_metrics["nmn"]["answer_accuracy"]

        # Super method will perform learning rate scheduling, serialize checkpoint, and log all
        # the validation metrics to tensorboard.
        super().after_validation(val_metrics, iteration)
