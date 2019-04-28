from typing import Any, Dict, Generator, List, Optional

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.utils.checkpointing import CheckpointManager


class _Trainer(object):
    """A base class for generic trainer which can have multiple models interacting with each
    other. An implementation of a class extending this trainer will contain the core training
    loop logic. This base class offers full flexibility, with sensible defaults which may be
    changed or disabled while extending this class.

    1. Default Adam Optimizer, this optimizer updates parameters of all models passed to this
       trainer in constructor. Learning rate and weight decay for this optimizer are picked up
       from the provided config.

    2. Default `ReduceLROnPlateau` learning rate scheduler. Gamma and patience arguments
       are picked up from the provided config. Threshold is 1e-2, and the observed metric is
       assumed to be of type "higher is better". In opposite case (e.g. val loss), make sure to
       reciprocate (or negate) the observed metric.

    3. Tensorboard logging of loss curves, metrics etc.

    4. Serialization of `models` and `optimizers` as checkpoint (.pth) files, after validation.
       The observed metric for keeping track of best checkpoint is of type "higher is better",
       follow (2) above if the observed metric is of type "lower is better".

    Note
    ----
    All models are "passed by assignment", so they could be used seamlessly in a separate
    ``_Evaluator``. Do not set `self._models` attribute like `self._models = ...` anywhere while
    overriding this class. Setting dict elements should be fine as `self._models` dict is mutable.
    """

    def __init__(
        self,
        config: Config,
        dataloader: DataLoader,
        models: Dict[str, nn.Module],
        serialization_dir: str,
        gpu_ids: List[int] = [0],
    ):
        self._C = config

        # Make dataloader cyclic for sampling batches perpetually.
        self._dataloader = self._cycle(dataloader)
        self._models = models

        # Set device according to specified GPU ids.
        self._device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids[0] >= 0 else "cpu")

        # Shift models to device, and wrap in DataParallel for Multi-GPU execution (if needed).
        for model_name in self._models:
            self._models[model_name] = self._models[model_name].to(self._device)

            if len(gpu_ids) > 1 and -1 not in gpu_ids:
                # Don't wrap to DataParallel if single GPU ID or -1 (CPU) is provided.
                self._models[model_name] = nn.DataParallel(self._models[model_name], gpu_ids)

        # Accumulate parameters of all models to construct Adam Optimizer.
        all_parameters: List[Any] = []
        for model_name in self._models:
            all_parameters.extend(list(self._models[model_name].parameters()))
        self._optimizer = optim.Adam(
            all_parameters, lr=self._C.OPTIM.LR_INITIAL, weight_decay=self._C.OPTIM.WEIGHT_DECAY
        )

        # Default learning rate scheduler: (lr *= gamma) when observed metric plateaus for
        # "patience" number of validation steps.
        self._lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="max",
            factor=self._C.OPTIM.LR_GAMMA,
            patience=self._C.OPTIM.LR_PATIENCE,
            threshold=1e-3,
        )

        # Tensorboard summary writer for logging losses and metrics.
        self._tensorboard_writer = SummaryWriter(log_dir=serialization_dir)
        self._checkpoint_manager = CheckpointManager(
            serialization_dir=serialization_dir,
            models=self._models,
            optimizer=self._optimizer,
            mode="max",
            filename_prefix=self._C.PHASE,
        )

        # Initialize a counter to keep track of the iteration number.
        # This increments everytime ``step`` is called.
        self._iteration: int = 0

    @property
    def iteration(self):
        return self._iteration

    @property
    def models(self):
        return self._models

    def step(self, iteration: Optional[int] = None):
        self._before_iteration()

        batch = next(self._dataloader)
        output_dict = self._do_iteration(batch)
        self._after_iteration(output_dict)

        self._iteration = iteration or self._iteration + 1

    def _before_iteration(self):
        self._optimizer.zero_grad()

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        # What a single iteration usually would look like.
        iteration_output_dict = self._models["model"](batch)
        batch_loss = iteration_output_dict["loss"].mean()
        batch_loss.backward()

        return {"loss": batch_loss}

    def _after_iteration(self, output_dict: Dict[str, Any], iteration: Optional[int] = None):
        self._optimizer.step()

        # keys: {"loss"} + ... {other keys such as "elbo"}
        for key in output_dict:
            if isinstance(output_dict[key], dict):
                # Use `add_scalars` for dicts in a nested `output_dict`.
                self._tensorboard_writer.add_scalars(
                    f"train/{key}", output_dict[key], self._iteration
                )
            else:
                # Use `add_scalar` for floats / zero-dim tensors in `output_dict`.
                self._tensorboard_writer.add_scalar(
                    f"train/{key}", output_dict[key], self._iteration
                )

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):
        if iteration is not None:
            self._iteration = iteration

        # Serialize model and optimizer and keep track of best checkpoint.
        # Add negative sign with perplexity to make it "higher is better".
        self._checkpoint_manager.step(val_metrics["metric"], self._iteration)

        # Perform learning rate scheduling based on validation perplexity.
        # Add negative sign with perplexity to make it "higher is better".
        self._lr_scheduler.step(val_metrics["metric"])

        # Log learning rate after scheduling.
        self._tensorboard_writer.add_scalar(
            "train/lr", self._optimizer.param_groups[0]["lr"], self._iteration
        )

        # Log all validation metrics to tensorboard.
        # For `ProgramPrior`, keys: {"perplexity"}
        # For `ProgramGenerator`, keys: {"BLEU", "perplexity", "sequence_accuracy"}
        # For `QuestionReconstructor`, keys: {"BLEU", "perplexity", "sequence_accuracy"}
        # For `NeuralModuleNetwork`, keys: {"answer_accuracy"}
        val_metrics.pop("metric")
        for model_name in val_metrics:
            for metric_name in val_metrics[model_name]:
                self._tensorboard_writer.add_scalar(
                    f"val/metrics/{model_name}/{metric_name}",
                    val_metrics[model_name][metric_name],
                    self._iteration,
                )

    def load_checkpoint(self, checkpoint_path: str, iteration: Optional[int] = None):
        training_checkpoint: Dict[str, Any] = torch.load(checkpoint_path)
        for key in training_checkpoint:
            if key == "optimizer":
                self._optimizer.load_state_dict(training_checkpoint[key])
            else:
                self._models[key].load_state_dict(training_checkpoint[key])

        # Infer iteration number from checkpoint file name, if not specified.
        self._iteration = iteration or int(checkpoint_path.split("_")[-1][:-4])

    def _cycle(self, dataloader: DataLoader) -> Generator[Dict[str, torch.Tensor], None, None]:
        """A generator which yields a random batch from dataloader perpetually. This generator is
        used in the constructor.

        This is done so because we train for a fixed number of iterations, and do not have the
        notion of 'epochs'. Using itertools.cycle with dataloader is harmful and may cause
        unexpeced memory leaks.
        """
        while True:
            for batch in dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self._device)
                yield batch
