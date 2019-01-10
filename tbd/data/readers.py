"""
A Reader simply reads data from disk and returns it almost as is. Readers should be utilized by
PyTorch ``Dataset``s. Any type of data pre-processing is not recommended in the reader, such as
tokenizing words to integers, embedding tokens, or passing an image through a pre-trained CNN.

Each reader must implement three methods:
    - ``__len__`` to return the length of data this Reader can read.
    - ``__getitem__`` to return data based on a unique index (which behaves as a primary key).
    - ``keys`` to return a list of possible primary keys (or indices) this Reader can provide
      data of.
"""
import h5py
import torch


class ClevrTokensReader(object):
    """
    A Reader for retrieving tokenized CLEVR programs, questions and answers, and corresponding
    image indices from a single HDF file containing this pre-processed data.

    Parameters
    ----------
    clevr_hdfpath: str
        Path to an HDF file containing tokenized programs, questions, answers and corresponding
        image indices.
    """

    def __init__(self, clevr_hdfpath: str):
        # questions, image indices, programs, and answers are small enough to load into memory
        with h5py.File(clevr_hdfpath, "r") as clevr_tokens:
            self.programs = torch.LongTensor(clevr_tokens["programs"][:])
            self.questions = torch.LongTensor(clevr_tokens["questions"][:])
            self.answers = torch.LongTensor(clevr_tokens["answers"][:])
            self.image_indices = torch.LongTensor(clevr_tokens["image_indices"][:])
            self._split = clevr_tokens.attrs["split"]

    def __len__(self):
        return self.image_indices.size(0)

    def __getitem__(self, index):
        program = self.programs[index]
        question = self.questions[index]
        answer = self.answers[index]
        image_index = self.image_indices[index]

        return {
            "program": program, "question": question, "answer": answer, "image_index": image_index
        }

    @property
    def split(self):
        return self._split
