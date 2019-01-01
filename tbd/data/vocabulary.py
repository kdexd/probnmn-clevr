"""
A Vocabulary maintains a mapping between words and corresponding unique integers, holds special
integers (tokens) for indicating start and end of sequence, and offers functionality to map
out-of-vocabulary words to the corresponding token.
"""
from typing import Dict, List, Union


class Vocabulary(object):
    """
    A simple Vocabulary class which maintains a mapping between words and integer tokens. In this
    codebase, it shall represent vocabularies of program, questions and answers of CLEVR. It can
    either be initialized by a list of unique words or a pre-saved vocabulary mapping with words
    as keys and integers as values (refer ``mapping`` property).

    Note: Padding index is 0 by default, compatible with PyTorch modules like ``nn.Embedding`` and
          several utils like ``pad_sequence``.

    Parameters
    ----------
    unique_words: List[str]
        A list of unique words belonging to either programs, questions or answers of CLEVR v1.0
        training dataset.
    """

    SOS_TOKEN = "<START>"
    EOS_TOKEN = "<END>"
    UNK_TOKEN = "<UNK>"

    SOS_INDEX = 1
    EOS_INDEX = 2
    UNK_INDEX = 3

    def __init__(self, unique_words: List[str] = []):
        self.word2index = {}
        self.word2index[""] = 0
        self.word2index[self.SOS_TOKEN] = self.SOS_INDEX
        self.word2index[self.EOS_TOKEN] = self.EOS_INDEX
        self.word2index[self.UNK_TOKEN] = self.UNK_INDEX
        for index, word in enumerate(unique_words):
            self.word2index[word] = index + 4

        self.index2word = {index: word for word, index in self.word2index.items()}

    @property
    def mapping(self) -> Dict[str, int]:
        return self.word2index

    @mapping.setter
    def mapping(self, mapping: Dict[str, int]) -> None:
        self.word2index = mapping
        self.index2word = {index: word for word, index in self.word2index.items()}

    def to_indices(self, words: List[str]) -> List[int]:
        return [self.word2index.get(word, self.UNK_INDEX) for word in words]

    def to_words(self, indices: List[int]) -> List[str]:
        return [self.index2word.get(index, self.UNK_TOKEN) for index in indices]

    def __len__(self):
        return len(self.index2word)
