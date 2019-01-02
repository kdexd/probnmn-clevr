"""
A checkpoint manager periodically saves model and optimizer as .pth files during training.

It also helps with experiment reproducibility in two ways. First, it saves the experiment
config file in checkpoint directory, so the saved checkpoints can be associated with all
their hyper-parameters. Second, it records the commit of codebase in the checkpoint directory
and later raises a warning if the checkpoints are loaded with another version of codebase.
This way, it gives a signal to view commit diffs for potential bugs, if something is fishy.
"""
import copy
import os
import shutil
from subprocess import PIPE, Popen
from typing import Optional, Tuple, Type
import warnings

import torch
from torch import nn, optim


class CheckpointManager(object):
    """
    A checkpoint manager which accepts path to an empty / non-existent directory and references
    to model and optimizer objects, and saves checkpoints every ``step_size`` epochs. Closely
    follows the API of PyTorch optimizers and learning rate schedulers.

    Parameters
    ----------
    checkpoint_dirpath: str
        Path to an empty or non-existent directory to save checkpoints.
    model: nn.Module
        Model which needs to be checkpointed.
    optimizer: optim.Optimizer
        Corresponding optimizer which needs to be checkpointed.
    metric_mode: str, optional (default="max")
        One of min, max. In min mode, best checkpoint will be recorded when metric hits a lower
        value; in max mode it will be recorded when metric hits a higher value.
    step_size: int, optional (default=1)
        Period of saving checkpoints (in terms of epochs).
    last_epoch: int, optional (default=-1)
        The index of last epoch. Usually -1 if training starts afresh, else depends on the epoch
        from where training is resumed.
    """

    def __init__(self,
                 checkpoint_dirpath: str,
                 config_ymlpath: str,
                 model: Type[nn.Module],
                 optimizer: Type[optim.Optimizer],
                 metric_mode: str = "max",
                 step_size: int = 1,
                 last_epoch: int = -1):
        self.checkpoint_dirpath = checkpoint_dirpath
        self.model = model
        self.optimizer = optimizer
        self.metric_mode = metric_mode

        # frequency of saving checkpoints (based on number of calls to ``step`` method)
        self.step_size = step_size
        self.last_epoch = last_epoch

        # initialize members to hold best checkpoint and its performance
        self.metric = None
        self.best_checkpoint = copy.copy(self._get_model_state_dict())

        # create checkpoint directory and copy the config in it
        os.makedirs(checkpoint_dirpath, exist_ok=True)
        shutil.copy(config_ymlpath, checkpoint_dirpath)

        # save current git commit hash in this checkpoint directory
        commit_sha_subprocess = Popen(
            ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
        )
        commit_sha, _ = commit_sha_subprocess.communicate()
        with open(os.path.join(checkpoint_dirpath, "commit_sha.txt"), "w") as commit_sha_file:
            commit_sha_file.write(commit_sha.decode("utf-8").strip().replace("\n", ""))

    def step(self, metric: Type[torch.Tensor]):
        """
        Save checkpoint if step size conditions meet, and update best checkpoint based on
        metric and metric mode.

        Parameters
        ----------
        metric: torch.Tensor
            Zero-dimensional tensor (scalar) holding the metric.
        """

        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0:
            save_state_dict = {
                "model": self._get_model_state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(
                save_state_dict,
                os.path.join(self.checkpoint_dirpath, f"model_{self.last_epoch}.pth"),
            )

        # update best checkpoint based on metric and metric mode
        if not self.metric:
            self.metric = metric

        if (self.metric_mode == "min" and metric < self.metric) or \
                (self.metric_mode == "max" and metric > self.metric):
            self.metric = metric
            self.best_checkpoint = copy.copy(self._get_model_state_dict())

    def save_best(self):
        """Save best performing checkpoint observed so far."""
        torch.save(
            self.best_checkpoint,
            os.path.join(self.checkpoint_dirpath, f"model_best_{self.metric.item()}.pth")
        )

    def _get_model_state_dict(self):
        if isinstance(self.model, nn.DataParallel):
            return self.model.module.state_dict()
        else:
            return self.model.state_dict()


def load_checkpoint(checkpoint_dirpath: str, epoch: int) -> Tuple[nn.Module, optim.Optimizer]:
    """
    Given a path to directory containing saved checkpoints and epoch number, load corresponding
    checkpoint. This method checks if current commit SHA of code matches the commit SHA recorded
    when this checkpoint was saved - raises a warning if they don't match.

    Parameters
    ----------
    checkpoint_dirpath: str
        Path to directory containing saved checkpoints (as created by ``create_checkpoint_dir``).
    epoch: int
        Epoch number for which checkpoint is to be loaded.

    Returns
    -------
    Tuple[nn.Module, optim.Optimizer]
        Model and optimizer stated dicts from checkpoint.
    """

    # verify commit sha, raise warning if it doesn't match
    current_commit_sha_subprocess = Popen(
        ["git", "rev-parse", "--short", "HEAD"], stdout=PIPE, stderr=PIPE
    )
    current_commit_sha, _ = current_commit_sha_subprocess.communicate()
    current_commit_sha = current_commit_sha.decode("utf-8").strip().replace("\n", "")

    with open(os.path.join(checkpoint_dirpath, "commit_sha.txt"), "r") as commit_sha_file:
        checkpoint_commit_sha = commit_sha_file.read().strip().replace("\n", "")

    if current_commit_sha != checkpoint_commit_sha:
        warnings.warn(
            f"Current commit ({current_commit_sha}) and the commit "
            f"({checkpoint_commit_sha}) from which checkpoint was saved,"
            " are different. This might affect reproducibility and results."
        )

    # derive checkpoint name / path from the epoch number
    checkpoint_pthpath = os.path.join(checkpoint_dirpath, f"model_{epoch}.pth")

    # load encoder, decoder, optimizer state_dicts
    components = torch.load(checkpoint_pthpath)
    return components["model"], components["optimizer"]
