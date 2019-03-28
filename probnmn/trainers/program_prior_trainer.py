import argparse
import logging
from typing import Any, Dict, Optional

from allennlp.data import Vocabulary
import torch
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ProgramPriorDataset
from probnmn.models import ProgramPrior
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class ProgramPriorTrainer(_Trainer):
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

        if self._C.PHASE != "program_prior":
            raise ValueError(
                f"Trying to initialize a ProgramPriorTrainer, expected config PHASE to be "
                f"program_prior, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = ProgramPriorDataset(self._A.tokens_train_h5)
        dataloader = DataLoader(dataset, batch_size=self._C.OPTIM.BATCH_SIZE, shuffle=True)

        # This will be a part of `self._models`, keep this handle for convenience.
        self._program_prior = ProgramPrior(
            vocabulary=Vocabulary.from_files(self._A.vocab_dirpath),
            input_size=self._C.PROGRAM_PRIOR.INPUT_SIZE,
            hidden_size=self._C.PROGRAM_PRIOR.HIDDEN_SIZE,
            num_layers=self._C.PROGRAM_PRIOR.NUM_LAYERS,
            dropout=self._C.PROGRAM_PRIOR.DROPOUT,
        ).to(device)

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={"program_prior": self._program_prior},
            device=device,
            serialization_dir=self._A.save_dirpath,
            start_iteration=start_iteration,
        )

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass and compute loss. Return an output dict
        which would be passed to `after_iteration`.
        """

        # Forward and backward passes through program_prior.
        # keys: {"predictions", "loss"}
        iteration_output_dict = self._program_prior(batch["program"])
        batch_loss = iteration_output_dict["loss"].mean()

        # `batch_loss` does not require grad only when printing qualitative examples.
        if batch_loss.requires_grad:
            batch_loss.backward()

            # Clamp all gradients between (-5, 5).
            for parameter in self._program_prior.parameters():
                if parameter.grad is not None:
                    parameter.grad.clamp_(min=-5, max=5)

        return {"loss": batch_loss}

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):

        # Set "metric" key in `val_metrics`, this governs learning rate scheduling and keeping
        # track of best checkpoint. For program prior, it will be perplexity (lower is better).
        # Reciprocate perplexity to make it "higher is better".
        val_metrics["metric"] = 1.0 / val_metrics["program_prior"]["perplexity"]

        # Super method will perform learning rate scheduling, serialize checkpoint, and log all
        # the validation metrics to tensorboard.
        super().after_validation(val_metrics, iteration)
