from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from tbd.data.readers import ClevrTokensReader


class ProgramPriorDataset(Dataset):
    """
    Provides programs as tokenized sequences to train the ``ProgramPrior``.

    Parameters
    ----------
    tokens_hdfpath: str
        Path to an HDF file to initialize the underlying reader.
    """

    def __init__(self, tokens_hdfpath: str):
        self._reader = ClevrTokensReader(tokens_hdfpath)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, index):
        # only return programs, nothing else needed for training program prior
        # also, return a dict for the sake of uniformity in return type of several classes
        return {
            "program": torch.tensor(self._reader[index]["program"]).long()
        }

    @property
    def split(self):
        return self._reader.split


class QuestionCodingDataset(Dataset):
    """
    Provides questions and programs as tokenized sequences for question coding. It also provides
    a "supervision" flag, which can behave as a maskwhen batched, to tune the amount of program
    supervision on ``ProgramGenerator``.

    Parameters
    ----------
    tokens_hdfpath: str
        Path to an HDF file to initialize the underlying reader.
    num_supervision: int, optional (default = None)
        Number of examples where there would be a program supervision over questions, for
        ``ProgramGenerator``.
    """

    def __init__(self,
                 tokens_hdfpath: str,
                 num_supervision: int = 699989):
        self._tokens = ClevrTokensReader(tokens_hdfpath)

        self._supervision_list = np.zeros(len(self._tokens))

        # 100% supervision by default, and there's no notion of supervision in val split.
        if self.split == "train" and num_supervision < len(self._tokens):
            # This would be completely deterministic if seed is set in training script.
            __supervision_examples = np.random.choice(
                list(range(self._supervision_list)), replace=False, size=num_supervision
            )
            self._supervision_list[__supervision_examples] = 1

        self._supervision_list = torch.tensor(self._supervision_list).long()

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, index):
        item = self._tokens[index]
        supervision = self._supervision_list[index]

        return {
            "program": torch.tensor(item["program"]).long(),
            "question": torch.tensor(item["question"]).long(),
            "supervision": supervision
        }

    @property
    def split(self):
        return self._tokens.split

    def get_supervision_list(self):
        """
        Return a list of 1's and 0's, indicating which examples have program supervision during
        question coding. It is used by ``tbd.data.sampler.SupervisionWeightedRandomSampler`` to
        form a mini-batch with nearly equal number of examples with/without program supervision.
        """
        return self._supervision_list
