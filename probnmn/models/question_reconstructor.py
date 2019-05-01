from allennlp.data import Vocabulary

from probnmn.config import Config
from probnmn.modules.seq2seq_base import Seq2SeqBase


class QuestionReconstructor(Seq2SeqBase):
    r"""
    A wrapper over :class:`probnmn.modules.seq2seq_base.Seq2SeqBase`. This sequence to sequence
    model accepts tokenized and padded program sequences and decodes them to question sequences.

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        AllenNLP's vocabulary. This vocabulary has three namespaces - "questions", "programs" and
        "answers", which contain respective token to integer mappings.
    input_size: int, optional (default = 256)
        The dimension of the inputs to the LSTM.
    hidden_size: int, optional (default = 256)
        The dimension of the outputs of the LSTM.
    num_layers: int, optional (default = 2)
        Number of recurrent layers in the LSTM.
    dropout: float, optional (default = 0.0)
        Dropout probability for the outputs of LSTM at each layer except last.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 input_size: int = 256,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.0):
        # 45 is max_program_length in CLEVR v1.0 train split.
        max_decoding_steps = 45

        super().__init__(
            vocabulary,
            source_namespace="programs",
            target_namespace="questions",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            max_decoding_steps=max_decoding_steps
        )

    @classmethod
    def from_config(cls, config: Config):
        r"""Instantiate this class directly from a :class:`~probnmn.config.Config`."""

        _C = config
        return cls(
            vocabulary=Vocabulary.from_files(_C.DATA.VOCABULARY),
            input_size=_C.QUESTION_RECONSTRUCTOR.INPUT_SIZE,
            hidden_size=_C.QUESTION_RECONSTRUCTOR.HIDDEN_SIZE,
            num_layers=_C.QUESTION_RECONSTRUCTOR.NUM_LAYERS,
            dropout=_C.QUESTION_RECONSTRUCTOR.DROPOUT,
        )
