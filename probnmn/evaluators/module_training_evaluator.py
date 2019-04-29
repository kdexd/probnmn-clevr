import logging
from typing import Any, Dict, List, Type

import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ModuleTrainingDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class ModuleTrainingEvaluator(_Evaluator):
    def __init__(
        self, config: Config, models: Dict[str, Type[nn.Module]], gpu_ids: List[int] = [0]
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
            dataset,
            batch_size=self._C.OPTIM.BATCH_SIZE,
            # num_workers=self._A.cpu_workers
        )

        super().__init__(config=config, dataloader=dataloader, models=models, gpu_ids=gpu_ids)

        # These will be a part of `self._models`, keep these handles for convenience.
        self._program_generator = self._models["program_generator"]
        self._nmn = self._models["nmn"]

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass to accumulate metrics in model objects."""

        sampled_programs = self._program_generator(batch["question"])["predictions"]
        output_dict = self._nmn(batch["image"], sampled_programs, batch["answer"])
        batch_loss = output_dict["loss"].mean()

        return {"loss": batch_loss}
