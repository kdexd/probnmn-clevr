from typing import Dict, List

from allennlp.data import Vocabulary
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Average
import torch
from torch import nn
from torch.nn import functional as F


class ProgramPrior(nn.Module):
    r"""
    A simple language model which learns a prior over all the valid program sequences in CLEVR
    v1.0 training split.

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

    def __init__(
        self,
        vocabulary: Vocabulary,
        input_size: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self._start_index = vocabulary.get_token_index("@start@", namespace="programs")
        self._end_index = vocabulary.get_token_index("@end@", namespace="programs")
        self._pad_index = vocabulary.get_token_index("@@PADDING@@", namespace="programs")
        self._unk_index = vocabulary.get_token_index("@@UNKNOWN@@", namespace="programs")

        vocab_size = vocabulary.get_vocab_size(namespace="programs")
        embedder_inner = Embedding(vocab_size, input_size, padding_index=self._pad_index)
        self._embedder = BasicTextFieldEmbedder({"programs": embedder_inner})

        self._encoder = PytorchSeq2SeqWrapper(
            nn.LSTM(
                input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
            )
        )
        # Project and tie input and output embeddings
        self._projection_layer = nn.Linear(hidden_size, input_size, bias=False)
        self._output_layer = nn.Linear(input_size, vocab_size, bias=False)
        self._output_layer.weight = embedder_inner.weight

        # Record average log2 (perplexity) for calculating final perplexity.
        self._log2_perplexity = Average()

    def forward(self, program_tokens: torch.Tensor):
        r"""
        Given tokenized program sequences padded upto maximum length, predict sequence at next
        time-step and calculate cross entropy loss of this predicted sequence.

        Parameters
        ----------
        program_tokens: torch.Tensor
            Tokenized program sequences padded with zeroes upto maximum length.
            shape: (batch_size, max_sequence_length)

        Returns
        -------
        Dict[str, torch.Tensor]
            Predictions of next time-step and cross entropy loss (by teacher forcing), a dict
            with structure::

                {
                    "predictions": torch.Tensor (shape: (batch_size, max_program_length - 1)),
                    "loss": torch.Tensor (shape: (batch_size, ))
                }
        """

        batch_size = program_tokens.size(0)

        # Add "@start@" and "@end@" tokens to program sequences.
        program_tokens, _ = add_sentence_boundary_token_ids(
            program_tokens, (program_tokens != self._pad_index), self._start_index, self._end_index
        )
        program_tokens_mask = (program_tokens != self._pad_index).long()
        # Excluding @start@ token, because this is used with output of LSTM (next time-step).
        program_lengths = program_tokens_mask[:, 1:].sum(-1).float()

        # shape: (batch_size, max_sequence_length, input_size)
        embedded_programs = self._embedder({"programs": program_tokens})

        # shape: (batch_size, max_sequence_length, hidden_size)
        encoded_programs = self._encoder(embedded_programs, program_tokens_mask)

        # shape: (batch_size, max_sequence_length, input_size)
        output_projection = self._projection_layer(encoded_programs)
        # shape: (batch_size, max_sequence_length, vocab_size)
        output_logits = self._output_layer(output_projection)

        output_class_probabilities = F.softmax(output_logits, dim=-1)
        # Don't sample @start@, @@PADDING@@ and @@UNKNOWN@@.
        output_class_probabilities[:, :, self._start_index] = 0
        output_class_probabilities[:, :, self._pad_index] = 0
        output_class_probabilities[:, :, self._unk_index] = 0

        batch_predictions: List[torch.Tensor] = []
        for batch_index in range(output_class_probabilities.size(0)):
            # Perform ancestral sampling instead of greedy decoding.
            # shape: (batch_size, )
            batch_index_predictions = torch.multinomial(
                output_class_probabilities[batch_index], 1
            ).squeeze()
            batch_predictions.append(batch_index_predictions)

        # shape: (batch_size, max_sequence_length)
        predictions = torch.stack(batch_predictions, 0)

        # Multiply with mask just to be sure.
        predictions = predictions[:, :-1] * program_tokens_mask[:, 1:]

        # shape: (batch_size, )
        sequence_cross_entropy = sequence_cross_entropy_with_logits(
            output_logits[:, :-1, :].contiguous(),
            program_tokens[:, 1:].contiguous(),
            weights=program_tokens_mask[:, 1:],
            average=None,
        )
        # Record metrics aggregated over current batch during evaluation.
        if not self.training:
            self._log2_perplexity(sequence_cross_entropy.mean().item())
        return {"predictions": predictions, "loss": sequence_cross_entropy}

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        r"""
        Return perplexity using the accumulated loss.

        Parameters
        ----------
        reset: bool, optional (default = True)
            Whether to reset the accumulated metrics after retrieving them.

        Returns
        -------
        Dict[str, float]
            A dictionary with metrics ``{"perplexity"}``.
        """

        return {"perplexity": 2 ** self._log2_perplexity.get_metric(reset=reset)}
