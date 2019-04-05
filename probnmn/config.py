r"""This module provides package-wide configuration management."""
from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    A collection of all the required configuration parameters. This class is a nested dict-like
    structure, with nested keys accessible as attributes. It contains sensible default values for
    all the parameters, which may be overriden by (first) through a YAML file and (second) through
    a list of attributes and values.

    Extended Summary
    ----------------
    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    Parameters
    ----------
    config_yaml: str
        Path to a YAML file containing configuration parameters to override.
    config_override: List[Any], optional (default= [])
        A list of sequential attributes and values of parameters to override. This happens after
        overriding from YAML file.

    Examples
    --------
    Let a YAML file named "config.yaml" specify these parameters to override::

        ALPHA: 1000.0
        BETA: 0.5

    >>> _C = Config("config.yaml", ["OPTIM.BATCH_SIZE", 2048, "BETA", 0.7])
    >>> _C.ALPHA  # default: 100.0
    1000.0
    >>> _C.BATCH_SIZE  # default: 256
    2048
    >>> _C.BETA  # default: 0.1
    0.7

    Attributes
    ----------
    RANDOM_SEED: 0
        Random seed for NumPy and PyTorch, important for reproducibility.

    PHASE: "joint_training"
        Which phase to train (or evaluate) on? One of ``program_prior``, ``question_coding``,
        ``module_training`` or ``joint_training``.

    SUPERVISION: 1000
        Number of training examples where questions have paired ground-truth programs. These
        examples are chosen randomly (no stochasticity for a fixe ``RANDOM_SEED``).

    SUPERVISION_QUESTION_MAX_LENGTH: 40
        Maximum length of questions to be considered for choosing ``SUPERVISION`` number of
        training examples. Longer questions will not have paired ground-truth programs by default.

    OBJECTIVE: "ours"
        Training objective, ``baseline`` - only use ``SUPERVISION`` examples for training.
        truth programs, and ``ours`` - use the whole dataset for training.
    __________

    DATA:
        Collection of required data paths for training and evaluation. All these are assumed to be
        relative to project root directory. If elsewhere, symlinking is recommended.

    DATA.VOCABULARY: "clevr_vocabulary"
        Path to a directory containing CLEVR v1.0 vocabulary (readable by AllenNLP).

    DATA.TRAIN_TOKENS: "data/clevr_train_tokens.h5"
        Path to H5 file containing tokenized programs, questions and answers, and corresponding
        image indices for CLEVR v1.0 train split.

    DATA.TRAIN_FEATURES: "data/clevr_train_features.h5"
        Path to H5 file containing pre-extracted features from CLEVR v1.0 train images.

    DATA.VAL_TOKENS: "data/clevr_val_tokens.h5"
        Path to H5 file containing tokenized programs, questions and answers, and corresponding
        image indices for CLEVR v1.0 val split.

    DATA.VAL_FEATURES: "data/clevr_val_features.h5"
        Path to H5 file containing pre-extracted features from CLEVR v1.0 val images.
    __________

    PROGRAM_PRIOR:
        Parameters controlling the model architecture of Program Prior (LSTM language model).

    PROGRAM_PRIOR.INPUT_SIZE: 256
        The dimension of the inputs to the LSTM.

    PROGRAM_PRIOR.HIDDEN_SIZE: 256
        The dimension of the outputs of the LSTM.

    PROGRAM_PRIOR.NUM_LAYERS: 2
        Number of recurrent layers in the LSTM.

    PROGRAM_PRIOR.DROPOUT: 0.0
        Dropout probability for the outputs of LSTM at each layer except last.
    __________

    PROGRAM_GENERATOR:
        Parameters controlling the model architecture of Program Generator (Seq2Seq model). Here,
        the model encodes questions and decodes programs.

    PROGRAM_GENERATOR.INPUT_SIZE: 256
        The dimension of the inputs to the encoder and decoder.

    PROGRAM_GENERATOR.HIDDEN_SIZE: 256
        The dimension of the outputs of the encoder and decoder.

    PROGRAM_GENERATOR.NUM_LAYERS: 2
        Number of recurrent layers in the LSTM.

    PROGRAM_GENERATOR.DROPOUT: 0.0
        Dropout probability for the outputs of LSTM at each layer except last.
    __________

    QUESTION_RECONSTRUCTOR:
        Parameters controlling the model architecture of Question Reconstructor (Seq2Seq model).
        Here, the model encodes programs and decodes questions.

    QUESTION_RECONSTRUCTOR.INPUT_SIZE: 256
        The dimension of the inputs to the encoder and decoder.

    QUESTION_RECONSTRUCTOR.HIDDEN_SIZE: 256
        The dimension of the outputs of the encoder and decoder.

    QUESTION_RECONSTRUCTOR.NUM_LAYERS: 2
        Number of recurrent layers in the LSTM.

    QUESTION_RECONSTRUCTOR.DROPOUT: 0.0
        Dropout probability for the outputs of LSTM at each layer except last.
    __________

    NMN:
        Parameters controlling the model architecture of Neural Module Network. Here, the model
        takes an image and a program, lays out a pipeline of neural modules and executes it to
        get an answer.

    NMN.IMAGE_FEATURE_SIZE: [1024, 14, 14]
        Shape of input image features, in the form (channel, height, width).

    NMN.MODULE_CHANNELS: 128
        Number of channels for each neural module's convolutional blocks.

    NMN.CLASS_PROJECTION_CHANNELS: 1024
        Number of channels in projected final feature map (input to classifier).

    NMN.CLASSIFIER_LINEAR_SIZE: 1024
        Size of input to the classifier.
    __________

    ALPHA: 100.0
        Supervision scaling co-efficient. The negative log-likelihood loss of program generation
        and question reconstruction for examples with (GT) program supervision is scaled by this
        factor. Used during question coding and joint training.

    BETA: 0.1
        KL co-efficient. KL-divergence in ELBO is scaled by this factor. Used during question
        coding and joint training.

    GAMMA: 1.0
        Answer log-likelihood scaling co-efficient during joint training.

    DELTA: 0.99
        Decay co-efficient for moving average REINFORCE baseline. Used during question coding and
        joint training.
    __________

    OPTIM:
        Optimization hyper-parameters, relevant during training a particular phase.

    OPTIM.BATCH_SIZE: 256
        Batch size during training and evaluation.

    OPTIM.NUM_ITERATIONS: 20000
        Number of iterations to train for, batches are randomly sampled.

    OPTIM.WEIGHT_DECAY: 0.0
        Weight decay co-efficient for the optimizer.

    OPTIM.LR_INITIAL: 0.00001
        Initial learning rate for :class:`torch.optim.lr_scheduler.ReduceLROnPlateau`.

    OPTIM.LR_GAMMA: 0.5
        Factor to scale learning rate when an observed metric plateaus.

    OPTIM.LR_PATIENCE: 3
        Number of validation steps to wait and observe improvement in observed metric, before
        reducing the learning rate.
    __________

    CHECKPOINTS:
        Paths to pre-trained checkpoints of a particular phase to be used in subsequent phases.

    CHECKPOINTS.PROGRAM_PRIOR: "checkpoints/program_prior_best.pth"
        Path to pre-trained Program Prior checkpoint. Used during question coding.

    CHECKPOINTS.QUESTION_CODING: "checkpoints/question_coding_1000_baseline_best.pth"
        Path to pre-trained question coding checkpoint containing Program Prior (unchanged from
        ``program_prior`` phase), Program generator and Question Reconstructor. Used during
        module training and joint training.

    CHECKPOINTS.MODULE_TRAINING: "checkpoints/module_training_1000_baseline_best.pth"
        Path to pre-trained question coding checkpoint containing Program Generator (unchanged
        from ``question_coding`` phase) and Neural Module Network. Used during joint training.
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        self._C = CN()
        self._C.RANDOM_SEED = 0

        self._C.PHASE = "joint_training"
        self._C.SUPERVISION = 1000
        self._C.SUPERVISION_QUESTION_MAX_LENGTH = 40
        self._C.OBJECTIVE = "ours"

        self._C.DATA = CN()
        self._C.DATA.VOCABULARY = "data/clevr_vocabulary"

        self._C.DATA.TRAIN = CN()
        self._C.DATA.TRAIN_TOKENS = "data/clevr_train_tokens.h5"
        self._C.DATA.TRAIN_FEATURES = "data/clevr_train_features.h5"

        self._C.DATA.VAL = CN()
        self._C.DATA.VAL_TOKENS = "data/clevr_val_tokens.h5"
        self._C.DATA.VAL_FEATURES = "data/clevr_val_features.h5"

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
        self._C.GAMMA = 1.0
        self._C.DELTA = 0.99

        self._C.OPTIM = CN()
        self._C.OPTIM.BATCH_SIZE = 256
        self._C.OPTIM.NUM_ITERATIONS = 20000
        self._C.OPTIM.WEIGHT_DECAY = 0.0

        self._C.OPTIM.LR_INITIAL = 0.00001
        self._C.OPTIM.LR_GAMMA = 0.5
        self._C.OPTIM.LR_PATIENCE = 3

        self._C.CHECKPOINTS = CN()
        self._C.CHECKPOINTS.PROGRAM_PRIOR = "checkpoints/program_prior_best.pth"
        self._C.CHECKPOINTS.QUESTION_CODING = "checkpoints/question_coding_1000_ours_best.pth"
        self._C.CHECKPOINTS.MODULE_TRAINING = "checkpoints/module_training_1000_ours_best.pth"

        # Override parameter values from YAML file first, then from override list.
        self._C.merge_from_file(config_yaml)
        self._C.merge_from_list(config_override)

        # Make an instantiated object of this class immutable.
        self._C.freeze()

    def dump(self, file_path: str):
        r"""Save config at the specified file path.

        Parameters
        ----------
        file_path: str
            (YAML) path to save config at.
        """
        self._C.dump(stream=open(file_path, "w"))

    def __getattr__(self, attr: str):
        return self._C.__getattr__(attr)

    def __str__(self):
        return _config_str(self)

    def __repr__(self):
        return self._C.__repr__()


def _config_str(config: Config) -> str:
    r"""
    Collect a subset of config in sensible order (not alphabetical) according to phase. Used by
    :func:`Config.__str__()`.

    Parameters
    ----------
    config: Config
        A :class:`Config` object which is to be printed.
    """
    _C = config

    __C: CN = CN({"PHASE": _C.PHASE, "RANDOM_SEED": _C.RANDOM_SEED})
    common_string: str = str(__C) + "\n"

    if _C.PHASE in {"question_coding", "joint_training"}:
        __C = CN()  # type: ignore
        __C.OBJECTIVE = _C.OBJECTIVE
        __C.SUPERVISION = _C.SUPERVISION
        __C.SUPERVISION_QUESTION_MAX_LENGTH = _C.SUPERVISION_QUESTION_MAX_LENGTH
        common_string += str(__C) + "\n"

    common_string += str(_C.DATA) + "\n"

    if _C.PHASE in {"program_prior", "question_coding", "joint_training"}:
        common_string += str(CN({"PROGRAM_PRIOR": _C.PROGRAM_PRIOR})) + "\n"

    if _C.PHASE in {"question_coding", "module_training", "joint_training"}:
        common_string += str(CN({"PROGRAM_GENERATOR": _C.PROGRAM_GENERATOR})) + "\n"

    if _C.PHASE in {"question_coding", "joint_training"}:
        common_string += str(CN({"QUESTION_RECONSTRUCTOR": _C.QUESTION_RECONSTRUCTOR})) + "\n"

    if _C.PHASE in {"module_training", "joint_training"}:
        common_string += str(CN({"NMN": _C.NMN})) + "\n"

    if _C.PHASE in {"question_coding", "joint_training"}:
        __C = CN()  # type: ignore
        __C.ALPHA = _C.ALPHA
        __C.BETA = _C.BETA
        __C.DELTA = _C.DELTA
        if _C.PHASE == "joint_training":
            __C.GAMMA = _C.GAMMA
        common_string += str(__C) + "\n"

    common_string += str(CN({"OPTIM": _C.OPTIM})) + "\n"

    if _C.PHASE == "question_coding":
        __C = CN()
        __C.CHECKPOINTS = CN({"PROGRAM_PRIOR": _C.CHECKPOINTS.PROGRAM_PRIOR})
    elif _C.PHASE == "module_training":
        __C = CN()
        __C.CHECKPOINTS = CN({"QUESTION_CODING": _C.CHECKPOINTS.QUESTION_CODING})
    elif _C.PHASE == "joint_training":
        __C = CN()
        __C.CHECKPOINTS = CN()
        __C.CHECKPOINTS.QUESTION_CODING = _C.CHECKPOINTS.QUESTION_CODING
        __C.CHECKPOINTS.MODULE_TRAINING = _C.CHECKPOINTS.MODULE_TRAINING
    else:
        __C = CN()

    common_string += str(__C) + "\n"
    return common_string
