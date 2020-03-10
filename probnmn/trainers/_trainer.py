from typing import Any, Dict, Generator, List, Optional

from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from probnmn.config import Config
from probnmn.utils.checkpointing import CheckpointManager


class _Trainer(object):
    r"""
    A base class for generic training of models. This class can have multiple models interacting
    with each other, rather than a single model, which is suitable to our use-case (for example,
    ``module_training`` phase has two models:
    :class:`~probnmn.models.program_generator.ProgramGenerator` and
    :class:`~probnmn.models.nmn.NeuralModuleNetwork`). It offers full flexibility, with sensible
    defaults which may be changed (or disabled) while extending this class.

    Extended Summary
    ----------------
    1. Default :class:`~torch.optim.Adam` Optimizer, updates parameters of all models in this
       trainer. Learning rate and weight decay for this optimizer are picked up from the provided
       config.

    2. Default :class:`~torch.optim.lr_scheduler.ReduceLROnPlateau` learning rate scheduler. Gamma
       and patience arguments are picked up from the provided config. Observed metric is assumed
       to be of type "higher is better". For 'lower is better" metrics, make sure to reciprocate.

    3. Tensorboard logging of loss curves, metrics etc.

    4. Serialization of models and optimizer as checkpoint (.pth) files after every validation.
       The observed metric for keeping track of best checkpoint is of type "higher is better",
       follow (2) above if the observed metric is of type "lower is better".

    Extend this class and override suitable methods as per requirements, some important ones are:

    1. :meth:`step`, provides complete customization, this is the method which comprises of one
       full training iteration, and internally calls (in order) - :meth:`_before_iteration`,
       :meth:`_do_iteration` and :meth:`_after_iteration`. Most of the times you may not require
       overriding this method, instead one of the mentioned three methods called by `:meth:`step`.

    2. :meth:`_do_iteration`, with core training loop - what happens every iteration, given a
       ``batch`` from the dataloader this class holds.

    3. :meth:`_before_iteration` and :meth:`_after_iteration`, for any pre- or post-processing
       steps. Default behaviour:

        * :meth:`_before_iteration` - call ``optimizer.zero_grad()``
        * :meth:`_after_iteration` - call ``optimizer.step()`` and do tensorboard logging.

    4. :meth:`after_validation`, to specify any steps after evaluation. Default behaviour is to
       do learning rate scheduling and log validation metrics on tensorboard.

    Notes
    -----
    All models are `passed by assignment`, so they could be shared with an external evaluator.
    Do not set ``self._models = ...`` anywhere while extending this class.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    dataloader: torch.utils.data.DataLoader
        A :class:`~torch.utils.data.DataLoader` which provides batches of training examples. It
        wraps one of :mod:`probnmn.data.datasets` depending on the evaluation phase.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other during training. These are one or more from
        :mod:`probnmn.models` depending on the training phase.
    serialization_dir: str
        Path to a directory for tensorboard logging and serializing checkpoints.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.
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

        # Checkpoint manager to serialize model, optimizer and lr scheduler periodically.
        self._checkpoint_manager = CheckpointManager(
            serialization_dir=serialization_dir,
            keep_recent=100,
            optimizer=self._optimizer,
            scheduler=self._lr_scheduler,
            **models,
        )
        # Initialize a counter to keep track of the iteration number.
        # This increments everytime ``step`` is called.
        self._iteration: int = -1

    def step(self, iteration: Optional[int] = None):
        r"""
        Perform one iteration of training.

        Parameters
        ----------
        iteration: int, optional (default = None)
            Iteration number (useful to hard set to any number when loading checkpoint).
            If ``None``, use the internal :attr:`self._iteration` counter.
        """
        self._before_iteration()

        batch = next(self._dataloader)
        output_dict = self._do_iteration(batch)
        self._after_iteration(output_dict)

        self._iteration = iteration or self._iteration + 1

    def _before_iteration(self):
        r"""
        Steps to do before doing the forward pass of iteration. Default behavior is to simply
        call :meth:`zero_grad` for optimizer. Called inside :meth:`step`.
        """
        self._optimizer.zero_grad()

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Forward and backward passes on models, given a batch sampled from dataloader.

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of training examples sampled from dataloader. See :func:`step` and
            :meth:`_cycle` on how this batch is sampled.

        Returns
        -------
        Dict[str, Any]
            An output dictionary typically returned by the models. This would be passed to
            :meth:`_after_iteration` for tensorboard logging.
        """
        # What a single iteration usually would look like.
        iteration_output_dict = self._models["model"](batch)
        batch_loss = iteration_output_dict["loss"].mean()
        batch_loss.backward()
        return {"loss": batch_loss}

    def _after_iteration(self, output_dict: Dict[str, Any]):
        r"""
        Steps to do after doing the forward pass of iteration. Default behavior is to simply
        do gradient update through ``optimizer.step()``, and log metrics to tensorboard.

        Parameters
        ----------
        output_dict: Dict[str, Any]
            This is exactly the object returned by :meth:_do_iteration`, which would contain all
            the required losses for tensorboard logging.
        """
        self._optimizer.step()

        # keys: {"loss"} + ... {other keys such as "elbo"}
        for key in output_dict:
            if isinstance(output_dict[key], dict):
                # Use ``add_scalars`` for dicts in a nested ``output_dict``.
                self._tensorboard_writer.add_scalars(
                    f"train/{key}", output_dict[key], self._iteration
                )
            else:
                # Use ``add_scalar`` for floats / zero-dim tensors in ``output_dict``.
                self._tensorboard_writer.add_scalar(
                    f"train/{key}", output_dict[key], self._iteration
                )

    def after_validation(self, val_metrics: Dict[str, Any], iteration: Optional[int] = None):
        r"""
        Steps to do after an external :class:`~probnmn.evaluators._evaluator._Evaluator` performs
        evaluation. This is not called by :meth:`step`, call it from outside at appropriate time.
        Default behavior is to perform learning rate scheduling, serializaing checkpoint and to
        log validation metrics to tensorboard.

        Since this implementation assumes a key ``"metric"`` in ``val_metrics``, it is convenient
        to set this key while overriding this method, when there are multiple models and multiple
        metrics and there is one metric which decides best checkpoint.

        Parameters
        ----------
        val_metrics: Dict[str, Any]
            Validation metrics for all the models. Returned by ``evaluate`` method of
            :class:`~probnmn.evaluators._evaluator._Evaluator` (or its extended class).
        iteration: int, optional (default = None)
            Iteration number. If ``None``, use the internal :attr:`self._iteration` counter.
        """
        if iteration is not None:
            self._iteration = iteration

        # Serialize model and optimizer and keep track of best checkpoint.
        self._checkpoint_manager.step(self._iteration, val_metrics["metric"])

        # Perform learning rate scheduling based on validation perplexity.
        self._lr_scheduler.step(val_metrics["metric"])

        # Log learning rate after scheduling.
        self._tensorboard_writer.add_scalar(
            "train/lr", self._optimizer.param_groups[0]["lr"], self._iteration
        )

        # Log all validation metrics to tensorboard (pop the "metric" key, which was only relevant
        # to learning rate scheduling and checkpointing).
        val_metrics.pop("metric")
        for model_name in val_metrics:
            for metric_name in val_metrics[model_name]:
                self._tensorboard_writer.add_scalar(
                    f"val/metrics/{model_name}/{metric_name}",
                    val_metrics[model_name][metric_name],
                    self._iteration,
                )

    def load_checkpoint(self, checkpoint_path: str, iteration: Optional[int] = None):
        r"""
        Load a checkpoint to continue training from. The iteration when this checkpoint was
        serialized, is inferred from its name (so do not rename after serialization).

        Parameters
        ----------
        checkpoint_path: str
            Path to a checkpoint containing models and optimizers of the phase which is being
            trained on.

        iteration: int, optional (default = None)
            Iteration number. If ``None``, get it from the checkpoint.
        """
        _iteration = self._checkpoint_manager.load(checkpoint_path)

        # By default, the provided iteration overrides what is found in checkpoint.
        iteration = iteration or _iteration
        self._iteration = iteration

    def _cycle(self, dataloader: DataLoader) -> Generator[Dict[str, torch.Tensor], None, None]:
        r"""
        A generator which yields a random batch from dataloader perpetually. This generator is
        used in the constructor.

        Extended Summary
        ----------------
        This is done so because we train for a fixed number of iterations, and do not have the
        notion of 'epochs'. Using ``itertools.cycle`` with dataloader is harmful and may cause
        unexpeced memory leaks.
        """
        while True:
            for batch in dataloader:
                for key in batch:
                    batch[key] = batch[key].to(self._device)
                yield batch

    @property
    def iteration(self):
        return self._iteration

    @property
    def models(self):
        return self._models
