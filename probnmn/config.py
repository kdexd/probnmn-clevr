from yacs.config import CfgNode as CN
import yaml


_C = CN()
_C.PHASE = "program_prior"
_C.RANDOM_SEED = 0

_C.PROGRAM_PRIOR = CN()
_C.PROGRAM_PRIOR.INPUT_SIZE = 256
_C.PROGRAM_PRIOR.HIDDEN_SIZE = 256
_C.PROGRAM_PRIOR.NUM_LAYERS = 2
_C.PROGRAM_PRIOR.DROPOUT = 0.0

_C.OPTIM = CN()
_C.OPTIM.BATCH_SIZE = 256
_C.OPTIM.NUM_ITERATIONS = 10000
_C.OPTIM.WEIGHT_DECAY = 0.0

_C.OPTIM.LR_INITIAL = 0.01
_C.OPTIM.LR_GAMMA = 0.5
_C.OPTIM.LR_PATIENCE = 3


def get_config_defaults():
    return _C.clone()
