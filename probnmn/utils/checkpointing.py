"""A checkpoint manager periodically saves model and optimizer as .pth files during training, and
keeps track of best performing checkpoint based on a particular metric. Checkpoint managers help
with experiment reproducibility, they record the commit SHA of your current codebase in the
checkpoint saving directory. While loading any checkpoint from other commit, they raise a friendly
warning, a signal to inspect commit diffs for potential bugs.
"""
import copy
from pathlib import Path
from subprocess import PIPE, Popen
import warnings

import torch
from torch import nn, optim
import yaml


class CheckpointManager(object):
    """A checkpoint manager saves state dicts of model and optimizer as .pth files in a specified
    directory. This class closely follows the API of PyTorch optimizers and learning rate
    schedulers.

    Note::
        For ``DataParallel`` modules, ``model.module.state_dict()`` is
        saved, instead of ``model.state_dict()``.

    Parameters
    ----------
    model: nn.Module
        Wrapped model, which needs to be checkpointed.
    optimizer: optim.Optimizer
        Wrapped optimizer which needs to be checkpointed.
    checkpoint_dirpath: str
        Path to an empty or non-existent directory to save checkpoints.
    mode: str, optional (default="max")
        One of `min`, `max`. In `min` mode, best checkpoint will be
        recorded when metric hits a lower value; in `max` mode it will
        be recorded when metric hits a higher value.
    step_size: int, optional (default=1)
        Period of saving checkpoints.
    last_epoch: int, optional (default=-1)
        The index of last epoch.

    Example
    -------
    >>> model = torch.nn.Linear(10, 2)
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> ckpt_manager = CheckpointManager(model, optimizer, "/tmp/ckpt", mode="min")
    >>> for epoch in range(20):
    ...     train(...)
    ...     val_loss = validate(...)
    ...     ckpt_manager.step(val_loss)
    """

    def __init__(self, model, optimizer, checkpoint_dirpath, mode="max",
                 step_size=1, last_epoch=-1, filename_prefix="model", **kwargs):

        if not isinstance(model, nn.Module):
            raise TypeError("{} is not a Module".format(
                type(model).__name__))

        if not isinstance(optimizer, optim.Optimizer):
            raise TypeError("{} is not an Optimizer".format(
                type(optimizer).__name__))

        self.model = model
        self.optimizer = optimizer
        self.ckpt_dirpath = Path(checkpoint_dirpath)
        self.mode = mode
        self.step_size = step_size
        self.last_epoch = last_epoch
        self.filename_prefix = filename_prefix
        self.init_directory(**kwargs)

        # initialize members to hold best checkpoint and its performance
        self.best_metric = None
        self.best_ckpt = copy.copy(self._model_state_dict())

    def init_directory(self, config={}):
        """Initialize empty checkpoint directory and record commit SHA in it. Extend this method
        to do more fancy things at start of experiment, for example saving hyper-parameters as a
        JSON file.
        """

        self.ckpt_dirpath.mkdir(parents=True, exist_ok=True)
        # save current git commit hash in this checkpoint directory
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")
        commit_sha_filepath = self.ckpt_dirpath / f".commit-{commit_sha}"
        commit_sha_filepath.touch()
        yaml.dump(
            config, open(str(self.ckpt_dirpath / "config.yml"), "w"),
            default_flow_style=False
        )

    def step(self, metric, epoch=None):
        """Save checkpoint if step size conditions meet, and update best checkpoint based
        on metric and mode.
        """

        if not epoch:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # Update best checkpoint based on metric and metric mode.
        if not self.best_metric:
            self.best_metric = metric

        if (self.mode == "min" and metric < self.best_metric) or \
                (self.mode == "max" and metric > self.best_metric):
            self.best_metric = metric
            self.best_ckpt = copy.copy(self._model_state_dict())

        if not self.last_epoch % self.step_size:
            torch.save(
                {
                    "model": self._model_state_dict(),
                    "optimizer": self.optimizer.state_dict()
                },
                self.ckpt_dirpath / f"{self.filename_prefix}_{self.last_epoch}.pth",
            )

        # Save best performing checkpoint observed so far.
        torch.save(
            self.best_ckpt, self.ckpt_dirpath / f"{self.filename_prefix}_best.pth"
        )

    def _model_state_dict(self):
        """Returns state dict of model, taking care of DataParallel case."""
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_pthpath):
    """Given a path to saved checkpoint, load corresponding state dicts
    of model and optimizer from it. This method checks if the current
    commit SHA of codebase matches the commit SHA recorded when this
    checkpoint was saved by checkpoint manager.

    Parameters
    ----------
    checkpoint_pthpath: str or pathlib.Path
        Path to saved checkpoint (as created by ``CheckpointManager``).

    Returns
    -------
    nn.Module, optim.Optimizer
        Model and optimizer state dicts loaded from checkpoint.

    Raises
    ------
    UserWarning
        If commit SHA do not match, or if the directory doesn't have
        the recorded commit SHA.
    """

    if isinstance(checkpoint_pthpath, str):
        checkpoint_pthpath = Path(checkpoint_pthpath)
    checkpoint_dirpath = checkpoint_pthpath.resolve().parent
    checkpoint_commit_sha = list(checkpoint_dirpath.glob(".commit-*"))

    if len(checkpoint_commit_sha) == 0:
        warnings.warn("Commit SHA was not recorded while saving checkpoints.")
    else:
        # verify commit sha, raise warning if it doesn't match
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        commit_sha = commit_sha.decode("utf-8").strip().replace("\n", "")

        # remove ".commit-"
        checkpoint_commit_sha = checkpoint_commit_sha[0].name[8:]

        if commit_sha != checkpoint_commit_sha:
            warnings.warn(
                f"Current commit ({commit_sha}) and the commit "
                f"({checkpoint_commit_sha}) at which checkpoint was saved,"
                " are different. This might affect reproducibility."
            )
    components = torch.load(checkpoint_pthpath)

    if "model" in components:
        return components["model"], components["optimizer"]
    else:
        return components, None
