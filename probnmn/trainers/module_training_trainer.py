import argparse
import logging
from typing import Any, Dict, Optional

from allennlp.data import Vocabulary
import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ModuleTrainingDataset
from probnmn.models import NeuralModuleNetwork, ProgramGenerator
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class ModuleTrainingTrainer(_Trainer):
    def __init__(
        self,
        config: Config,
        args: argparse.Namespace,
        device: torch.device,
        start_iteration: Optional[int] = 0,
    ):
        self._C = config

        # TODO (kd): absorb args into Config.
        self._A = args

        if self._C.PHASE != "question_coding":
            raise ValueError(
                f"Trying to initialize a QuestionCodingTrainer, expected config PHASE to be "
                f"question_coding, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        self._vocabulary = Vocabulary.from_files(self._A.vocab_dirpath)

        dataset = ModuleTrainingDataset(
            self._A.tokens_train_h5, self._A.features_train_h5, in_memory=False
        )
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, num_workers=self._A.cpu_workers
        )

        # Vocabulary is needed to instantiate the models.
        vocabulary = Vocabulary.from_files(self._A.vocab_dirpath)

        # This will be a part of `self._models`, keep this handle for convenience.
        self._nmn = NeuralModuleNetwork(
            vocabulary=vocabulary,
            image_feature_size=tuple(self._C.NMN.IMAGE_FEATURE_SIZE),
            module_channels=self._C.NMN.MODULE_CHANNELS,
            class_projection_channels=self._C.NMN.CLASS_PROJECTION_CHANNELS,
            classifier_linear_size=self._C.NMN.CLASSIFIER_LINEAR_SIZE,
        ).to(device)

        if -1 not in self._A.gpu_ids:
            # Don't wrap to DataParallel for CPU-mode.
            self._nmn = nn.DataParallel(self._nmn, self._A.gpu_ids)

        # Program Generator checkpoint, this will be frozen during module training.
        self._program_generator = ProgramGenerator(
            vocabulary=vocabulary,
            input_size=self._C.PROGRAM_GENERATOR.INPUT_SIZE,
            hidden_size=self._C.PROGRAM_GENERATOR.HIDDEN_SIZE,
            num_layers=self._C.PROGRAM_GENERATOR.NUM_LAYERS,
            dropout=self._C.PROGRAM_GENERATOR.DROPOUT,
        ).to(device)

        self._program_generator.load_state_dict(
            torch.load(self._C.CHECKPOINTS.QUESTION_CODING)["program_generator"]
        )
        self._program_generator.eval()

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={"nmn": self._nmn, "program_generator": self._program_generator},
            device=device,
            serialization_dir=self._A.save_dirpath,
            start_iteration=start_iteration,
        )

        # Load NMN from saved checkpoint if specified.
        if self._A.checkpoint_pthpath != "":
            module_training_checkpoint = torch.load(self._A.checkpoint_pthpath)
            self._nmn.load_state_dict(module_training_checkpoint["nmn"])
            self._optimizer.load_state_dict(module_training_checkpoint["optimizer"])
            self._iteration = int(self._A.checkpoint_pthpath.split("_")[-1][:-4])

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
