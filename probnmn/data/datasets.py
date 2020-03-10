import numpy as np
import torch
from torch.utils.data import Dataset

from probnmn.data.readers import ClevrImageFeaturesReader, ClevrTokensReader


class ProgramPriorDataset(Dataset):
    r"""
    Provides programs as tokenized sequences to train the
    class:`~probnmn.models.program_prior.ProgramPrior`.

    Parameters
    ----------
    tokens_h5path: str
        Path to an HDF file to initialize the underlying reader.
    """

    def __init__(self, tokens_h5path: str):
        self._reader = ClevrTokensReader(tokens_h5path)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, index):
        # Only return programs, nothing else needed for training program prior.
        # Also, return a dict for the sake of uniformity in return type of several classes.
        return {"program": torch.tensor(self._reader[index]["program"]).long()}

    @property
    def split(self):
        return self._reader.split


class QuestionCodingDataset(Dataset):
    r"""
    Provides questions and programs as tokenized sequences for Question Coding. It also provides
    a "supervision" flag, which can behave as a mask when batched, to tune the amount of program
    supervision on :class:`~probnmn.models.program_generator.ProgramGenerator`.

    Notes
    -----
    For a fixed numpy random seed, the randomly generated supervision list will always be same.

    Parameters
    ----------
    tokens_h5path: str
        Path to an HDF file to initialize the underlying reader.
    num_supervision: int, optional (default = None)
        Number of examples where there would be a program supervision over questions, for
        :class:`~probnmn.models.program_generator.ProgramGenerator`.
    supervision_question_max_length: int, optional (default = 30)
        Maximum length of question for picking examples with program supervision.
    """

    def __init__(
        self,
        tokens_h5path: str,
        num_supervision: int = 699989,
        supervision_question_max_length: int = 40,
    ):
        self._tokens = ClevrTokensReader(tokens_h5path)

        self._supervision_list = np.zeros(len(self._tokens))

        # 100% supervision by default, and there's no notion of supervision in val split.
        if self.split == "train" and num_supervision < len(self._tokens):
            # Drop example indices where question length > max length.
            example_indices = np.ones(len(self._tokens))
            question_lengths = (self._tokens.questions != 0).sum(-1)
            example_indices[question_lengths > supervision_question_max_length] = 0
            example_indices = example_indices.nonzero()[0]

            # This would be completely deterministic if seed is set in training script.
            supervision_examples = np.random.choice(
                example_indices, replace=False, size=num_supervision
            )
            self._supervision_list[supervision_examples] = 1
        else:
            self._supervision_list += 1

        self._supervision_list = torch.tensor(self._supervision_list).long()

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        item = self._tokens[index]
        supervision = self._supervision_list[index]

        return {
            "program": torch.tensor(item["program"]).long(),
            "question": torch.tensor(item["question"]).long(),
            "supervision": supervision,
        }

    @property
    def split(self):
        return self._tokens.split

    def get_supervision_list(self):
        r"""
        Return a list of 1's and 0's, indicating which examples have program supervision during
        question coding. Used by :class:`~probnmn.data.samplers.SupervisionWeightedRandomSampler`
        to form a mini-batch with nearly equal number of examples with(out) program supervision.
        """
        return self._supervision_list


class ModuleTrainingDataset(Dataset):
    r"""
    Provides questions, image features an answers for module training. Programs are inferred by
    :class:`~probnmn.models.program_generator.ProgramGenerator` trained during Question Coding.

    Parameters
    ----------
    tokens_h5path: str
        Path to an HDF file to initialize the underlying reader.
    features_h5path: str
        Path to an HDF file containing a 'dataset' of pre-extracted image features.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(self, tokens_h5path: str, features_h5path: str, in_memory: bool = True):
        self._tokens = ClevrTokensReader(tokens_h5path)
        self._features = ClevrImageFeaturesReader(features_h5path, in_memory)

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        item = self._tokens[index]
        features = self._features[item["image_index"]]

        # We sample programs, but GT programs are needed during evaluation for computing metrics.
        return {
            "question": torch.tensor(item["question"]).long(),
            "answer": torch.tensor(item["answer"]).long(),
            "image": torch.tensor(features).float(),
            "program": torch.tensor(item["program"]).long(),
        }

    @property
    def split(self):
        return self._tokens.split


class JointTrainingDataset(Dataset):
    r"""
    Provides questions, programs, supervision flag, image features and answers for Joint Training.
    If the random seed is set carefully, then the supervision list is made same as that in
    :class:`QuestionCodingDataset`.

    Notes
    -----
    For a fixed numpy random seed, the randomly generated supervision list will always be same.

    Parameters
    ----------
    tokens_h5path: str
        Path to an HDF file to initialize the underlying reader.
    features_h5path: str
        Path to an HDF file containing a 'dataset' of pre-extracted image features.
    num_supervision: int, optional (default = None)
        Number of examples where there would be a program supervision over questions, for
        :class:`~probnmn.models.program_generator.ProgramGenerator`.
    supervision_question_max_length: int, optional (default = 30)
        Maximum length of question for picking examples with program supervision.
    in_memory: bool, optional (default = True)
        Whether to load all image features in memory.
    """

    def __init__(
        self,
        tokens_h5path: str,
        features_h5path: str,
        num_supervision: int = 699989,
        supervision_question_max_length: int = 30,
        in_memory: bool = True,
    ):

        self._tokens = ClevrTokensReader(tokens_h5path)
        self._features = ClevrImageFeaturesReader(features_h5path, in_memory)

        self._supervision_list = np.zeros(len(self._tokens))

        # 100% supervision by default, and there's no notion of supervision in val split.
        if self.split == "train" and num_supervision < len(self._tokens):
            # Drop example indices where question length > max length.
            example_indices = np.ones(len(self._tokens))
            question_lengths = (self._tokens.questions != 0).sum(-1)
            example_indices[question_lengths > supervision_question_max_length] = 0
            example_indices = example_indices.nonzero()[0]

            # This would be completely deterministic if seed is set in training script.
            supervision_examples = np.random.choice(
                example_indices, replace=False, size=num_supervision
            )
            self._supervision_list[supervision_examples] = 1
        else:
            self._supervision_list += 1

        self._supervision_list = torch.tensor(self._supervision_list).long()

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        item = self._tokens[index]
        features = self._features[item["image_index"]]
        supervision = self._supervision_list[index]

        if self.split == "test":
            return {
                # question_index is the primary ID, used to make the predictions JSON.
                "question_index": torch.tensor(index).long(),
                "question": torch.tensor(item["question"]).long(),
                "image": torch.tensor(features).float(),
            }
        else:
            return {
                "question": torch.tensor(item["question"]).long(),
                "answer": torch.tensor(item["answer"]).long(),
                "program": torch.tensor(item["program"]).long(),
                "image": torch.tensor(features).float(),
                "supervision": supervision,
            }

    @property
    def split(self):
        return self._tokens.split

    def get_supervision_list(self):
        r"""
        Return a list of 1's and 0's, indicating which examples have program supervision during
        question coding. Used by :class:`~probnmn.data.samplers.SupervisionWeightedRandomSampler`
        to form a mini-batch with nearly equal number of examples with(out) program supervision.
        """
        return self._supervision_list
