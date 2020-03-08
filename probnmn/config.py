from typing import Any, List

from yacs.config import CfgNode as CN


class Config(object):
    r"""
    This class provides package-wide configuration management. It is a nested dict-like structure
    with nested keys accessible as attributes. It contains sensible default values, which can be
    modified by (first) a YAML file and (second) a list of attributes and values.

    This class definition contains default values corresponding to ``joint_training`` phase, as it
    is the final training phase and uses almost all the configuration parameters. Modification of
    any parameter after instantiating this class is not possible, so you must override required
    parameter values in either through ``config_yaml`` file or ``config_override`` list.

    .. note::

        The instantiated object is "immutable" - any modification is prohibited. You must override
        required parameter values either through ``config_file`` or ``override_list``.

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
    """

    def __init__(self, config_yaml: str, config_override: List[Any] = []):

        _C = CN()

        # Random seed for NumPy and PyTorch, important for reproducibility.
        _C.RANDOM_SEED = 0

        # Which phase to train (or evaluate) on: one of "program_prior", "question_coding",
        # "module_training" or "joint_training".
        _C.PHASE = "joint_training"

        # Number of training examples where questions have paired ground-truth programs. These
        # examples are chosen randomly (no stochasticity for a fixed `RANDOM_SEED`).
        _C.SUPERVISION = 1000

        # Maximum length of questions to be considered for choosing subset of training examples.
        # Longer questions will not have paired ground-truth programs by default.
        _C.SUPERVISION_QUESTION_MAX_LENGTH = 40

        # Training objective:
        #   1. "baseline" - only use `SUPERVISION` examples for training (with programs).
        #   2. "ours" - use the whole dataset for training (including examples without programs).
        _C.OBJECTIVE = "ours"

        # ----------------------------------------------------------------------------------------
        #   Collection of required data paths for training and evaluation. All these are assumed
        #   to be relative to project root directory.
        # ----------------------------------------------------------------------------------------
        _C.DATA = CN()

        # Path to a directory containing CLEVR v1.0 vocabulary (readable by AllenNLP).
        _C.DATA.VOCABULARY = "data/clevr_vocabulary"

        _C.DATA.TRAIN = CN()
        _C.DATA.VAL = CN()
        _C.DATA.TEST = CN()

        # Path to H5 file containing tokenized programs, questions and answers, and corresponding
        # image indices for CLEVR v1.0 train split.
        _C.DATA.TRAIN_TOKENS = "data/clevr_train_tokens.h5"

        # Path to H5 file containing pre-extracted features from CLEVR v1.0 train images.
        _C.DATA.TRAIN_FEATURES = "data/clevr_train_features.h5"

        # Path to H5 file containing tokenized programs, questions and answers, and corresponding
        # image indices for CLEVR v1.0 val split.
        _C.DATA.VAL_TOKENS = "data/clevr_val_tokens.h5"

        # Path to H5 file containing pre-extracted features from CLEVR v1.0 val images.
        _C.DATA.VAL_FEATURES = "data/clevr_val_features.h5"

        # Path to H5 file containing tokenized questions, and corresponding image indices for CLEVR
        # v1.0 test split.
        _C.DATA.TEST_TOKENS = "data/clevr_test_tokens.h5"

        # Path to H5 file containing pre-extracted features from CLEVR v1.0 test images.
        _C.DATA.TEST_FEATURES = "data/clevr_test_features.h5"

        # ----------------------------------------------------------------------------------------
        #   Model architecture of Program Prior.
        # ----------------------------------------------------------------------------------------
        _C.PROGRAM_PRIOR = CN()

        # The dimension of the inputs to the LSTM.
        _C.PROGRAM_PRIOR.INPUT_SIZE = 256
        # The dimension of the outputs of the LSTM.
        _C.PROGRAM_PRIOR.HIDDEN_SIZE = 256
        # Number of recurrent layers in the LSTM.
        _C.PROGRAM_PRIOR.NUM_LAYERS = 2
        # Dropout probability for the outputs of LSTM at each layer except last.
        _C.PROGRAM_PRIOR.DROPOUT = 0.0

        # ----------------------------------------------------------------------------------------
        #   Model architecture of Program Generator.
        # ----------------------------------------------------------------------------------------
        _C.PROGRAM_GENERATOR = CN()

        # The dimension of the inputs to the encoder and decoder.
        _C.PROGRAM_GENERATOR.INPUT_SIZE = 256
        # The dimension of the outputs of the encoder and decoder.
        _C.PROGRAM_GENERATOR.HIDDEN_SIZE = 256
        # Number of recurrent layers in the LSTM.
        _C.PROGRAM_GENERATOR.NUM_LAYERS = 2
        # Dropout probability for the outputs of LSTM at each layer except last.
        _C.PROGRAM_GENERATOR.DROPOUT = 0.0

        # ----------------------------------------------------------------------------------------
        #   Model architecture of Question Reconstructor.
        # ----------------------------------------------------------------------------------------
        _C.QUESTION_RECONSTRUCTOR = CN()

        # The dimension of the inputs to the encoder and decoder.
        _C.QUESTION_RECONSTRUCTOR.INPUT_SIZE = 256
        # The dimension of the outputs of the encoder and decoder.
        _C.QUESTION_RECONSTRUCTOR.HIDDEN_SIZE = 256
        # Number of recurrent layers in the LSTM.
        _C.QUESTION_RECONSTRUCTOR.NUM_LAYERS = 2
        # Dropout probability for the outputs of LSTM at each layer except last.
        _C.QUESTION_RECONSTRUCTOR.DROPOUT = 0.0

        # ----------------------------------------------------------------------------------------
        #   Model architecture of Neural Module Network.
        # ----------------------------------------------------------------------------------------
        _C.NMN = CN()

        # Shape of input image features, in the form (channel, height, width).
        _C.NMN.IMAGE_FEATURE_SIZE = [1024, 14, 14]
        # Number of channels for each neural module's convolutional blocks.
        _C.NMN.MODULE_CHANNELS = 128
        # Number of channels in projected final feature map (input to classifier).
        _C.NMN.CLASS_PROJECTION_CHANNELS = 1024
        # Size of input to the classifier.
        _C.NMN.CLASSIFIER_LINEAR_SIZE = 1024

        # ----------------------------------------------------------------------------------------
        #   Loss function co-efficients (names as per paper equations).
        # ----------------------------------------------------------------------------------------

        # Supervision scaling co-efficient. The negative log-likelihood loss of program generation
        # and question reconstruction for examples with (GT) program supervision is scaled by this
        # factor. Used during question coding and joint training.
        _C.ALPHA = 100.0

        # KL co-efficient. KL-divergence in ELBO is scaled by this factor. Used during question
        # coding and joint training.
        _C.BETA = 0.1

        # Answer log-likelihood scaling co-efficient during joint training.
        _C.GAMMA = 1.0

        # Decay co-efficient for moving average REINFORCE baseline. Used during question coding and
        # joint training.
        _C.DELTA = 0.99

        # ----------------------------------------------------------------------------------------
        #   Optimization hyper-parameters, relevant during training a particular phase.
        #   Default: Adam optimizer, ReduceLRonPlateau.
        # ----------------------------------------------------------------------------------------
        _C.OPTIM = CN()

        # Batch size during training and evaluation.
        _C.OPTIM.BATCH_SIZE = 256
        # Number of iterations to train for, batches are randomly sampled.
        _C.OPTIM.NUM_ITERATIONS = 20000
        # Weight decay co-efficient for the optimizer.
        _C.OPTIM.WEIGHT_DECAY = 0.0

        # Initial learning rate for ReduceLROnPlateau.
        _C.OPTIM.LR_INITIAL = 0.00001
        # Factor to scale learning rate when an observed metric plateaus.
        _C.OPTIM.LR_GAMMA = 0.5
        # Number of validation steps to wait and observe improvement in observed metric, before
        # reducing the learning rate.
        _C.OPTIM.LR_PATIENCE = 3

        # ----------------------------------------------------------------------------------------
        #   Paths to pre-trained checkpoints of a particular phase to be used in subsequent phases.
        # ----------------------------------------------------------------------------------------
        _C.CHECKPOINTS = CN()

        # Program Prior checkpoint, used during question coding and joint training.
        _C.CHECKPOINTS.PROGRAM_PRIOR = "checkpoints/program_prior_best.pth"

        # Question coding checkpoint containing Program Prior (unchanged from "program_prior"
        # phase), Program Generator and Question Reconstructor. Used during module training and
        # joint training.
        _C.CHECKPOINTS.QUESTION_CODING = "checkpoints/question_coding_1000_ours_best.pth"

        # Module training checkpoint containing Program Generator (unchanged from "question_coding"
        # phase) and Neural Module Network. Used during joint training.
        _C.CHECKPOINTS.MODULE_TRAINING = "checkpoints/module_training_1000_ours_best.pth"

        # Override parameter values from YAML file first, then from override list.
        self._C = _C
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
        _C = self._C
        common_string: str = str(
            CN(
                {
                    "PHASE": _C.PHASE,
                    "RANDOM_SEED": _C.RANDOM_SEED,
                    "OBJECTIVE": _C.OBJECTIVE,
                    "SUPERVISION": _C.SUPERVISION,
                    "SUPERVISION_QUESTION_MAX_LENGTH": _C.SUPERVISION_QUESTION_MAX_LENGTH,
                }
            )
        ) + "\n"
        common_string += str(CN({"DATA": _C.DATA})) + "\n"
        common_string += str(CN({"PROGRAM_PRIOR": _C.PROGRAM_PRIOR})) + "\n"
        common_string += str(CN({"PROGRAM_GENERATOR": _C.PROGRAM_GENERATOR})) + "\n"
        common_string += str(CN({"QUESTION_RECONSTRUCTOR": _C.QUESTION_RECONSTRUCTOR})) + "\n"
        common_string += str(CN({"NMN": _C.NMN})) + "\n"
        common_string += str(
            CN(
                {
                    "ALPHA": _C.ALPHA,
                    "BETA": _C.BETA,
                    "GAMMA": _C.GAMMA,
                    "DELTA": _C.DELTA
                }
            )
        ) + "\n"
        common_string += str(CN({"OPTIM": _C.OPTIM})) + "\n"
        common_string += str(CN({"CHECKPOINTS": _C.CHECKPOINTS})) + "\n"
        return common_string

    def __repr__(self):
        return self._C.__repr__()
