from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.RANDOM_SEED = 0

        self._C.PHASE = "question_coding"
        self._C.OBJECTIVE = "ours"
        self._C.SUPERVISION = 1000
        self._C.SUPERVISION_QUESTION_MAX_LENGTH = 40

        self._C.PROGRAM_PRIOR = CN()
        self._C.PROGRAM_PRIOR.INPUT_SIZE = 256
        self._C.PROGRAM_PRIOR.HIDDEN_SIZE = 256
        self._C.PROGRAM_PRIOR.NUM_LAYERS = 2
        self._C.PROGRAM_PRIOR.DROPOUT = 0.0

        self._C.PROGRAM_GENERATOR = CN()
        self._C.PROGRAM_GENERATOR.INPUT_SIZE = 256
        self._C.PROGRAM_GENERATOR.HIDDEN_SIZE = 256
        self._C.PROGRAM_GENERATOR.NUM_LAYERS = 2
        self._C.PROGRAM_GENERATOR.DROPOUT = 0.0

        self._C.QUESTION_RECONSTRUCTOR = CN()
        self._C.QUESTION_RECONSTRUCTOR.INPUT_SIZE = 256
        self._C.QUESTION_RECONSTRUCTOR.HIDDEN_SIZE = 256
        self._C.QUESTION_RECONSTRUCTOR.NUM_LAYERS = 2
        self._C.QUESTION_RECONSTRUCTOR.DROPOUT = 0.0

        self._C.NMN = CN()
        self._C.NMN.IMAGE_FEATURE_SIZE = [1024, 14, 14]
        self._C.NMN.MODULE_CHANNELS = 128
        self._C.NMN.CLASS_PROJECTION_CHANNELS = 1024
        self._C.NMN.CLASSIFIER_LINEAR_SIZE = 1024

        self._C.ALPHA = 100.0
        self._C.BETA = 0.1
        self._C.DELTA = 0.99

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 256
        self._C.OPTIM.NUM_ITERATIONS = 60000
        self._C.OPTIM.WEIGHT_DECAY = 0.0

        self._C.OPTIM.LR_INITIAL = 0.001
        self._C.OPTIM.LR_GAMMA = 0.5
        self._C.OPTIM.LR_PATIENCE = 3

        self._C.CHECKPOINTS = CN()
        self._C.CHECKPOINTS.PROGRAM_PRIOR = "checkpoints/program_prior_best.pth"
        self._C.CHECKPOINTS.QUESTION_CODING = "checkpoints/question_coding_1000_ours_best.pth"

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
