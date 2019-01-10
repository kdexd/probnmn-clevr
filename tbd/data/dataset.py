import torch
from torch.utils.data import Dataset

from tbd.data.readers import ClevrTokensReader


class ProgramPriorDataset(Dataset):
    """
    Provides programs as tokenized sequences to train the ``ProgramPrior``.

    Parameters
    ----------
    tokens_hdfpath : str
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
    # TODO (kd): add limited supervision support after KL term in QuestionCoding works well

    def __init__(self, tokens_hdfpath: str):
        self._reader = ClevrTokensReader(tokens_hdfpath)

    def __len__(self):
        return len(self._reader)

    def __getitem__(self, index):
        item = self._reader[index]
        return {
            "program": torch.tensor(item["program"]).long(),
            "question": torch.tensor(item["question"]).long()
        }

    @property
    def split(self):
        return self._reader.split
