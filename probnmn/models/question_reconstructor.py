from allennlp.data import Vocabulary

from probnmn.models.seq2seq_base import Seq2SeqBase


class QuestionReconstructor(Seq2SeqBase):
    """
    Convenience wrapper over ``probnmn.models.seq2seq_base.Seq2SeqBase``. This Seq2Seq model
    accepts tokenized and padded program sequences and converts them to question sequences.

    # TODO (kd): can make a fancy encoder here and pass it to the super class.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 input_size: int = 256,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 average_loss_across_timesteps: bool = True,
                 average_logprobs_across_timesteps: bool = False):
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
            average_loss_across_timesteps=average_loss_across_timesteps,
            average_logprobs_across_timesteps=average_logprobs_across_timesteps,
            max_decoding_steps=__max_decoding_steps
        )

