from typing import Any, List

from yacs.config import CfgNode as CN
import yaml


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.RANDOM_SEED = 0

        self._C.PHASE = CN()
        self._C.PHASE.NAME = "program_prior"

        self._C.PROGRAM_PRIOR = CN()
        self._C.PROGRAM_PRIOR.INPUT_SIZE = 256
        self._C.PROGRAM_PRIOR.HIDDEN_SIZE = 256
        self._C.PROGRAM_PRIOR.NUM_LAYERS = 2
        self._C.PROGRAM_PRIOR.DROPOUT = 0.0

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 256
        self._C.OPTIM.NUM_ITERATIONS = 10000
        self._C.OPTIM.WEIGHT_DECAY = 0.0

        self._C.OPTIM.LR_INITIAL = 0.01
        self._C.OPTIM.LR_GAMMA = 0.5
        self._C.OPTIM.LR_PATIENCE = 3

        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)
        self._C.freeze()

    def dump(self, file_path: str):
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        return self._C.__str__()

    def __repr__(self):
        return self._C.__repr__()
