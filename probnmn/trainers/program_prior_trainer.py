import logging
from typing import Any, Dict, List, Optional

from allennlp.data import Vocabulary
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.data.datasets import ProgramPriorDataset
from probnmn.models import ProgramPrior
from ._trainer import _Trainer


logger: logging.Logger = logging.getLogger(__name__)


class ProgramPriorTrainer(_Trainer):
    def __init__(self, config: Config, serialization_dir: str, gpu_ids: List[int] = [0]):
        self._C = config

        if self._C.PHASE != "program_prior":
            raise ValueError(
                f"Trying to initialize a ProgramPriorTrainer, expected config PHASE to be "
                f"program_prior, found {self._C.PHASE}"
            )

        # Initialize dataloader and model.
        dataset = ProgramPriorDataset(self._C.DATA.TRAIN_TOKENS)
        dataloader = DataLoader(dataset, batch_size=self._C.OPTIM.BATCH_SIZE, shuffle=True)

        # This will be a part of `self._models`, keep this handle for convenience.
        program_prior = ProgramPrior(
            vocabulary=Vocabulary.from_files(self._C.DATA.VOCABULARY),
            input_size=self._C.PROGRAM_PRIOR.INPUT_SIZE,
            hidden_size=self._C.PROGRAM_PRIOR.HIDDEN_SIZE,
            num_layers=self._C.PROGRAM_PRIOR.NUM_LAYERS,
            dropout=self._C.PROGRAM_PRIOR.DROPOUT,
        )

        super().__init__(
            config=config,
            dataloader=dataloader,
            models={"program_prior": program_prior},
            serialization_dir=serialization_dir,
            gpu_ids=gpu_ids,
        )

        # This will be a part of `self._models`, keep this handle for convenience.
        self._program_prior = self._models["program_prior"]

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one iteration, take a forward pass and compute loss. Return an output dict
        which would be passed to `after_iteration`.
        """

        # Forward and backward passes through program_prior.
        # keys: {"predictions", "loss"}
        iteration_output_dict = self._program_prior(batch["program"])
        batch_loss = iteration_output_dict["loss"].mean()
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
