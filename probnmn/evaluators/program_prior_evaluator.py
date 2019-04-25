import argparse
import logging
from typing import Any, Dict, Optional, Type

from allennlp.data import Vocabulary
import torch
from torch import nn
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ProgramPriorDataset
from ._evaluator import _Evaluator


logger: logging.Logger = logging.getLogger(__name__)


class ProgramPriorEvaluator(_Evaluator):
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

        if self._C.PHASE != "program_prior":
            raise ValueError(
                f"Trying to initialize a {self.__name__}, expected config PHASE to be "
                f"program_prior, found {self._C.PHASE}"
            )

        # Initialize vocabulary, dataloader and model.
        self._vocabulary = Vocabulary.from_files(self._A.vocab_dirpath)

        dataset = ProgramPriorDataset(self._A.tokens_val_h5)
        dataloader = DataLoader(dataset, batch_size=self._C.OPTIM.BATCH_SIZE)

        super().__init__(config=config, dataloader=dataloader, models=models, device=device)

        # This will be a part of `self._models`, keep this handle for convenience.
        self._program_prior = self._models["program_prior"]

    def evaluate(self, num_batches: Optional[int] = None):
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
            output_dict = self._do_iteration(batch)

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

        return eval_metrics

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass to accumulate metrics in model objects."""

        # Forward pass through program_prior.
        # keys: {"predictions", "loss"}
        output_dict = self._program_prior(batch["program"])
        return output_dict
