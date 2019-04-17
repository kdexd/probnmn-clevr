from allennlp.data import Vocabulary

from probnmn.modules.seq2seq_base import Seq2SeqBase


class QuestionReconstructor(Seq2SeqBase):
    """Convenience wrapper over ``probnmn.models.seq2seq_base.Seq2SeqBase``. This Seq2Seq model
    accepts tokenized and padded program sequences and converts them to question sequences.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 input_size: int = 256,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.0):
        # 45 is max_program_length in CLEVR v1.0 train
        __max_decoding_steps = 45

        super().__init__(
            vocabulary,
            source_namespace="programs",
            target_namespace="questions",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            max_decoding_steps=__max_decoding_steps
        )

