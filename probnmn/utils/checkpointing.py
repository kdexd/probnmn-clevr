"""A checkpoint manager periodically serializes models and optimizer as .pth files during
training, and keeps track of best performing checkpoint based on a particular metric.
"""
import copy
from pathlib import Path
from typing import Any, Dict, Type

import torch
from torch import nn, optim


class CheckpointManager(object):
    """A checkpoint manager saves state dicts of models and optimizer as .pth files in a specified
    directory. This class closely follows the API of PyTorch optimizers and learning rate
    schedulers.

    Note
    ----
    For ``nn.DataParallel``, ``.module.state_dict()`` is called instead of ``.state_dict()``.

    Parameters
    ----------
    models: Dict[str, Type[nn.Module]]
        Models which need to be serialized as a checkpoint.
    optimizer: optim.Optimizer
        Optimizer which needs to be serialized as a checkpoint.
    serialization_dir: str
        Path to an empty or non-existent directory to save checkpoints.
    mode: str, optional (default="max")
        One of `min`, `max`. In `min` mode, best checkpoint will be recorded when metric hits a
        lower value; in `max` mode it will be recorded when metric hits a higher value.
    filename_prefix: str, optional (default="checkpoint")
        Prefix of the to-be-saved checkpoint files.

    Example
    -------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager({"model": model}, optimizer, "/tmp/ckpt", mode="min")
    >>> for epoch in range(20):
    ...     train(...)
    ...     val_loss = validate(...)
    ...     ckpt_manager.step(val_loss)
    """

    def __init__(
        self,
        models: Dict[str, Type[nn.Module]],
        optimizer: Type[optim.Optimizer],
        serialization_dir: str,
        mode: str = "max",
        filename_prefix: str = "checkpoint",
    ):
        for key in models:
            if not isinstance(models[key], nn.Module):
                raise TypeError("{} is not a Module".format(type(models).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))

        self._models = models
        self._optimizer = optimizer
        self._serialization_dir = Path(serialization_dir)

        self._mode = mode
        self._filename_prefix = filename_prefix

        # Initialize members to hold best checkpoint and its performance.
        self._best_metric = None
        self._best_ckpt = copy.copy(self._models_state_dict())

    def step(self, metric, epoch_or_iteration):
        """Save checkpoint if step size conditions meet, and update best checkpoint based
        on metric and mode.
        """

        # Update best checkpoint based on metric and metric mode.
        if not self._best_metric:
            self._best_metric = metric

        if (self._mode == "min" and metric < self._best_metric) or (
            self._mode == "max" and metric > self._best_metric
        ):
            self._best_metric = metric
            self._best_ckpt = copy.copy(self._models_state_dict())

        # Save checkpoint corresponding to current iteration.
        torch.save(
            {**self._models_state_dict(), "optimizer": self._optimizer.state_dict()},
            self._serialization_dir / f"{self._filename_prefix}_{epoch_or_iteration}.pth",
        )
        # Save best performing checkpoint observed so far.
        torch.save(self._best_ckpt, self._serialization_dir / f"{self._filename_prefix}_best.pth")

    def _models_state_dict(self):
        """Returns state dict of models, taking care of DataParallel case."""
        models_state_dict: Dict[str, Any] = {}
        for key in self._models:
            if isinstance(self._models[key], nn.DataParallel):
                models_state_dict[key] = self._models[key].module.state_dict()
            else:
                models_state_dict[key] = self._models[key].state_dict()

        return models_state_dict
