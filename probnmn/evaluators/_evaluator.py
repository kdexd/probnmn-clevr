from typing import Any, Dict, List, Optional, Type

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from probnmn.config import Config


class _Evaluator(object):
    r"""
    A base class for generic evaluation of models. This class can have multiple models interacting
    with each other, rather than a single model, which is suitable to our use-case (for example,
    ``module_training`` phase has two models: :class:`~probnmn.models.ProgramGenerator` and
    :class:`~probnmn.models.nmn.NeuralModuleNetwork`). It offers full flexibility, with sensible
    defaults which may be changed (or disabled) while extending this class.

    Extended Summary
    ----------------
    Extend this class and override :meth:`_do_iteration` method, with core evaluation loop - what
    happens every iteration, given a ``batch`` from the dataloader this class holds.

    Notes
    -----
    1. All models are `passed by assignment`, so they could be shared with an external trainer.
       Do not set ``self._models = ...`` anywhere while extending this class.

    2. An instantiation of this class will always be paired in conjunction to a
       :class:`~probnmn.trainers._trainer._Trainer`. Pass the models of trainer class while
       instantiating this class.

    Parameters
    ----------
    config: Config
        A :class:`~probnmn.Config` object with all the relevant configuration parameters.
    dataloader: torch.utils.data.DataLoader
        A :class:`~torch.utils.data.DataLoader` which provides batches of evaluation examples. It
        wraps one of :mod:`probnmn.data.datasets` depending on the evaluation phase.
    models: Dict[str, Type[nn.Module]]
        All the models which interact with each other for evaluation. These are one or more from
        :mod:`probnmn.models` depending on the evaluation phase.
    gpu_ids: List[int], optional (default=[0])
        List of GPU IDs to use or evaluation, ``[-1]`` - use CPU.
    """

    def __init__(
        self,
        config: Config,
        dataloader: DataLoader,
        models: Dict[str, Type[nn.Module]],
        gpu_ids: List[int] = [0],
    ):
        self._C = config
        self._dataloader = dataloader
        self._models = models

        # Set device according to specified GPU ids. This device is only required for batches,
        # models will already be on apropriate device already, if passed from trainer.
        self._device = torch.device(f"cuda:{gpu_ids[0]}" if gpu_ids[0] >= 0 else "cpu")

    @property
    def models(self):
        return self._models

    def evaluate(self, num_batches: Optional[int] = None) -> Dict[str, Any]:
        r"""
        Perform evaluation using first ``num_batches`` of dataloader and return all evaluation
        metrics from the models.

        Parameters
        ----------
        num_batches: int, optional (default=None)
            Number of batches to use from dataloader. If ``None``, use all batches.

        Returns
        -------
        Dict[str, Any]
            Final evaluation metrics for all the models.
        """

        # Switch all models to "eval" mode.
        for model_name in self._models:
            self._models[model_name].eval()

        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(self._dataloader, desc="validation")):
                for key in batch:
                    batch[key] = batch[key].to(self._device)

                _ = self._do_iteration(batch)
                if num_batches is not None and iteration > num_batches:
                    break

        # keys: `self._models.keys()`
        eval_metrics: Dict[str, Dict[str, Any]] = {}
        for model_name in self._models:

            # Get metrics recorded by a particular model. This `hasattr` check exists because
            # it is a generic base class, all the models in `probnmn.models` implement a
            # `get_metrics` method.
            if hasattr(self._models[model_name], "get_metrics"):
                # keys: names of metrics recorded by corresponding model.
                eval_metrics[model_name] = self._models[model_name].get_metrics()

            elif isinstance(self._models[model_name], nn.DataParallel):
                if hasattr(self._models[model_name].module, "get_metrics"):
                    eval_metrics[model_name] = self._models[model_name].module.get_metrics()

        # Switch all models back to "train" mode.
        for model_name in self._models:
            self._models[model_name].train()

        return eval_metrics

    def _do_iteration(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Core evaluation logic for one iteration, operates on a batch. This base class has a dummy
        implementation - just forward pass through some "model".

        Parameters
        ----------
        batch: Dict[str, Any]
            A batch of evaluation examples sampled from dataloader. See :func:`evaluate` on how
            this batch is sampled.

        Returns
        -------
        Dict[str, Any]
            An output dictionary typically returned by the models. This may contain predictions
            from models, validation loss etc.
        """

        output_dict = self._models["model"](batch)
        return output_dict
