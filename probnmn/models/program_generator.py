from allennlp.data import Vocabulary

from probnmn.models.seq2seq_base import Seq2SeqBase


class ProgramGenerator(Seq2SeqBase):
    """
    Convenience wrapper over ``probnmn.models.seq2seq_base.Seq2SeqBase``. This Seq2Seq model
    accepts tokenized and padded question sequences and converts them to program sequences.

    # TODO (kd): can make a fancy encoder here and pass it to the super class.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 dropout: float = 0.0,
                 max_decoding_steps: int = 26):  # 26 is max_program_length in CLEVR v1.0 train
        super().__init__(
            vocabulary,
            source_namespace="questions",
            target_namespace="programs",
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout=dropout,
            max_decoding_steps=max_decoding_steps
        )
