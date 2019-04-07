from allennlp.data import Vocabulary

from probnmn.modules.seq2seq_base import Seq2SeqBase


class ProgramGenerator(Seq2SeqBase):
    r"""
    A wrapper over :class:`probnmn.modules.seq2seq_base.Seq2SeqBase`. This sequence to sequence
    model accepts tokenized and padded question sequences and decodes them to program sequences.

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

        # 26 is max_program_length in CLEVR v1.0 train split.
        max_decoding_steps = 26

        super().__init__(
            vocabulary,
            source_namespace="questions",
            target_namespace="programs",
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            max_decoding_steps=max_decoding_steps
        )
