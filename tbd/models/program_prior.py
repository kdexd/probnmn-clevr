from typing import Dict, Type

from allennlp.data import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import sequence_cross_entropy_with_logits
import torch
from torch import nn


class ProgramPriorVanillaLSTM(nn.Module):
    """
    A simple language model with one-layer Vanilla LSTM, and no dropout. It
    learns a prior over all the valid program sequences in CLEVR v1.0 training
    split. This one is here to compare any added bells and whistles with this
    simple model, and make sure that it's not an overkill.
    
    Parameters
    ----------
    vocabulary: Vocabulary
        Vocabulary with namespaces for CLEVR programs, questions and answers.
        We'll only use the `programs` namespace though.
    embedding_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self._vocabulary = vocabulary

        # short-hand variables
        __PAD = self._vocabulary.get_token_index("__PAD__", namespace="programs")
        __vocab_size = self._vocabulary.get_vocab_size(namespace="programs")
        __embedder_inner = Embedding(__vocab_size, embedding_size, padding_index=__PAD)

        self._embedder = BasicTextFieldEmbedder({"programs": __embedder_inner})
        self._seq_encoder = PytorchSeq2SeqWrapper(
            nn.LSTM(
                embedding_size, hidden_size, num_layers=num_layers,
                dropout=dropout, batch_first=True
            )
        )
        # project and tie input and output embeddings
        self._projection_layer = nn.Linear(hidden_size, embedding_size, bias=False)
        self._output_layer = nn.Linear(embedding_size, __vocab_size, bias=False)
        self._output_layer.weight = __embedder_inner.weight

    def forward(self, program_tokens: torch.LongTensor):
        """
        Given tokenized program sequences padded upto maximum length, predict
        sequence at next time-step and calculate cross entropy loss of this
        predicted sequence.

        Parameters
        ----------
        program_tokens: torch.LongTensor
            Tokenized program sequences padded with zeroes upto maximum length.
            Shape: (batch_size, max_sequence_length)

        Returns
        -------
        Dict[str, Type[torch.Tensor]]
            A dict with two keys - `predicted_tokens` and `loss`.

            - `predicted_tokens` represents program sequences predicted for
              next time-step, shape: (batch_size, max_sequence_length - 1).

            - `loss` represents per sequence cross entropy loss, shape:
              (batch_size, )
        """

        # assume zero padding by default
        program_tokens_mask = (program_tokens != 0).long()
        # shape: (batch_size, max_sequence_length, embedding_size)
        embedded_programs = self._embedder({"programs": program_tokens})
        # shape: (batch_size, max_sequence_length, hidden_size)
        encoded_programs = self._seq_encoder(embedded_programs, program_tokens_mask)
        # shape: (batch_size, max_sequence_length, embedding_size)
        output_projection = self._projection_layer(encoded_programs)
        # shape: (batch_size, max_sequence_length, vocab_size)
        output_logits = self._output_layer(output_projection)

        _, next_timestep = torch.max(output_logits, dim=-1)
        # multiply with mask just to be sure
        next_timestep = next_timestep[:, :-1] * program_tokens_mask[:, 1:]

        sequence_cross_entropy = sequence_cross_entropy_with_logits(
            output_logits[:, :-1, :].contiguous(),
            program_tokens[:, 1:].contiguous(),
            weights=program_tokens_mask[:, 1:],
            average=None
        )
        output_dict = {
            "predicted_tokens": next_timestep,
            "loss": sequence_cross_entropy * program_tokens_mask[:, 1:].sum(-1).float()
        }
        return output_dict


class ProgramPriorResidualLSTM(nn.Module):
    """
    Language model with a two-layer LSTM with a residual connection from first
    layer to second layer.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 dropout: float = 0.0):
        super().__init__()
        self._vocabulary = vocabulary

        # short-hand variables
        __PAD = self._vocabulary.get_token_index("__PAD__", namespace="programs")
        __vocab_size = self._vocabulary.get_vocab_size(namespace="programs")
        __embedder_inner = Embedding(__vocab_size, embedding_size, padding_index=__PAD)

        self._embedder = BasicTextFieldEmbedder({"programs": __embedder_inner})
        self._seq_encoder_1 = nn.LSTM(
            embedding_size, hidden_size, batch_first=True
        )
        self._projection_layer_1 = nn.Linear(hidden_size, embedding_size, bias=False)

        self._dropout = nn.Dropout(dropout)

        self._seq_encoder_2 = nn.LSTM(
            embedding_size, hidden_size, batch_first=True
        )
        self._projection_layer_2 = nn.Linear(hidden_size, embedding_size, bias=False)

        self._output_layer = nn.Linear(embedding_size, __vocab_size, bias=False)
        self._output_layer.weight = __embedder_inner.weight

        self._seq_encoder_1 = PytorchSeq2SeqWrapper(self._seq_encoder_1)
        self._seq_encoder_2 = PytorchSeq2SeqWrapper(self._seq_encoder_2)

    def forward(self, program_tokens: torch.LongTensor):
        # assume zero padding by default
        program_tokens_mask = (program_tokens != 0).long()
        # shape: (batch_size, max_sequence_length, embedding_size)
        embedded_programs = self._embedder({"programs": program_tokens})
        # shape: (batch_size, max_sequence_length, hidden_size)
        encoded_programs_1 = self._seq_encoder_1(embedded_programs, program_tokens_mask)
        # shape: (batch_size, max_sequence_length, embedding_size)
        projection_1 = self._projection_layer_1(encoded_programs_1)
        dropout_projection_1 = self._dropout(projection_1)

        # shape: (batch_size, max_sequence_length, hidden_size)
        encoded_programs_2 = self._seq_encoder_2(dropout_projection_1, program_tokens_mask)
        # shape: (batch_size, max_sequence_length, embedding_size)
        projection_2 = self._projection_layer_2(encoded_programs_1)

        # shape: (batch_size, max_sequence_length, embedding_size)
        residual = projection_1 + projection_2
        # shape: (batch_size, max_sequence_length, vocab_size)
        output_logits = self._output_layer(residual)

        _, next_timestep = torch.max(output_logits, dim=-1)
        # multiply with mask just to be sure
        next_timestep = next_timestep[:, :-1] * program_tokens_mask[:, 1:]

        sequence_cross_entropy = sequence_cross_entropy_with_logits(
            output_logits[:, :-1, :].contiguous(),
            program_tokens[:, 1:].contiguous(),
            weights=program_tokens_mask[:, 1:],
            average=None
        )
        output_dict = {
            "predicted_tokens": next_timestep,
            "loss": sequence_cross_entropy * program_tokens_mask[:, 1:].sum(-1).float()
        }
        return output_dict


# default program prior model used in all experiments
ProgramPrior = ProgramPriorResidualLSTM
