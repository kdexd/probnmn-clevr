import argparse
import logging
from typing import Any, Dict, Type

from allennlp.data import Vocabulary
import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import JointTrainingDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class JointTrainingEvaluator(_Evaluator):
    def __init__(
        self,
        config: Config,
        args: argparse.Namespace,
        models: Dict[str, Type[nn.Module]],
        device: torch.device,
    ):
        self._C = config

        # TODO (kd): absorb args into Config.
        self._A = args

        if self._C.PHASE != "joint_training":
            raise ValueError(
                f"Trying to initialize a JointTrainingEvaluator, expected config PHASE to be "
                f"joint_training, found {self._C.PHASE}"
            )

        # Initialize vocabulary, dataloader and model.
        self._vocabulary = Vocabulary.from_files(self._A.vocab_dirpath)

        # There is no notion of "supervision" during evaluation.
        dataset = JointTrainingDataset(self._A.tokens_val_h5, self._A.features_val_h5)
        dataloader = DataLoader(
            dataset, batch_size=self._C.OPTIM.BATCH_SIZE, num_workers=self._A.cpu_workers
        )

        super().__init__(config=config, dataloader=dataloader, models=models, device=device)

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._nmn = self._models["nmn"]

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass to accumulate metrics in model objects."""

        # Forward pass through program generator and neural module network.
        # keys: {"predictions", "loss"}
        pg_output_dict = self._program_generator(batch["question"], batch["program"])
        nmn_output_dict = self._nmn(batch["image"], pg_output_dict["predictions"], batch["answer"])

        return {"program_generator": pg_output_dict, "nmn": nmn_output_dict}
