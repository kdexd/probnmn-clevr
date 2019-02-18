from typing import Dict, List

from allennlp.data import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Average
import torch
from torch import nn
from torch.nn import functional as F


class ProgramPrior(nn.Module):
    """
    A simple language model which learns a prior over all the valid program sequences in CLEVR
    v1.0 training split.

    Parameters
    ----------
    vocabulary: Vocabulary
        Vocabulary with namespaces for CLEVR programs, questions and answers. We'll only use the
        `programs` namespace though.
    input_size: int
    hidden_size: int
    num_layers: int
    dropout: float
    average_loss_across_timesteps: bool, optional (default = True)
        Whether to average cross entropy loss (teacher forcing) across time-steps. If `False`,
        it will be summed across time-steps.
    average_logprobs_across_timesteps: bool, optional (default = False)
        Whether to average sampled sequence log-probabilities across time-steps. If ``False``,
        they will be summed across time-steps.
    """

    def __init__(self,
                 vocabulary: Vocabulary,
                 input_size: int = 256,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.0,
                 average_loss_across_timesteps: bool = True,
                 average_logprobs_across_timesteps: bool = False):
        super().__init__()
        self._vocabulary = vocabulary

        self._start_index = vocabulary.get_token_index("@start@", namespace="programs")
        self._end_index = vocabulary.get_token_index("@end@", namespace="programs")
        self._pad_index = vocabulary.get_token_index("@@PADDING@@", namespace="programs")
        self._unk_index = vocabulary.get_token_index("@@UNKNOWN@@", namespace="programs")

        vocab_size = self._vocabulary.get_vocab_size(namespace="programs")
        embedder_inner = Embedding(vocab_size, input_size, padding_index=self._pad_index)
        self._embedder = BasicTextFieldEmbedder({"programs": embedder_inner})

        self._encoder = nn.LSTM(
            input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True
        )

        # Project and tie input and output embeddings
        self._projection_layer = nn.Linear(hidden_size, input_size, bias=False)
        self._output_layer = nn.Linear(input_size, vocab_size, bias=False)
        self._output_layer.weight = embedder_inner.weight

        # Flags for averaging loss and log-probabilities across time-step.
        self._average_loss_across_timesteps = average_loss_across_timesteps
        self._average_logprobs_across_timesteps = average_logprobs_across_timesteps

        # Record average log2 (perplexity) for calculating final perplexity.
        self._log2_perplexity = Average()

    def forward(self, program_tokens: torch.LongTensor):
        """
        Given tokenized program sequences padded upto maximum length, predict sequence at next
        time-step and calculate cross entropy loss of this predicted sequence.

        Parameters
        ----------
        program_tokens: torch.LongTensor
            Tokenized program sequences padded with zeroes upto maximum length.
            Shape: (batch_size, max_sequence_length)

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict with two keys - `predicted_tokens` and `loss`.

            - `predicted_tokens` represents program sequences predicted for
              next time-step, shape: (batch_size, max_sequence_length - 1).

            - `loss` represents per sequence cross entropy loss, shape:
              (batch_size, )

              - `sequence_logprobs` represents sequence log probs, like beam search.
                It is essentially (-`loss`) * (program lengths). shape: (batch_size, )
        """

        batch_size = program_tokens.size(0)

        # Add "@start@" and "@end@" tokens to program sequences.
        program_tokens, _ = add_sentence_boundary_token_ids(
            program_tokens, (program_tokens != self._pad_index),
            self._start_index, self._end_index
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
        # Don't sample @start@, @@PADDING@@ and @@UNKNOWN@@
        output_class_probabilities[:, :, self._start_index] = 0
        output_class_probabilities[:, :, self._pad_index] = 0
        output_class_probabilities[:, :, self._unk_index] = 0

        # Perform ancestral sampling instead of greedy decoding
        batch_next_timestep: List[torch.Tensor] = []
        for batch_index in range(output_class_probabilities.size(0)):
            batch_index_next_timestep = torch.multinomial(
                output_class_probabilities[batch_index], 1).squeeze()
            batch_next_timestep.append(batch_index_next_timestep)

        # shape: (batch_size, max_sequence_length)
        next_timestep = torch.stack(batch_next_timestep, 0)

        # multiply with mask just to be sure
        next_timestep = next_timestep[:, :-1] * program_tokens_mask[:, 1:]

        # shape: (batch_size, )
        sequence_cross_entropy = sequence_cross_entropy_with_logits(
            output_logits[:, :-1, :].contiguous(),
            program_tokens[:, 1:].contiguous(),
            weights=program_tokens_mask[:, 1:],
            average=None
        )
        # We typically always do teacher forcing for our usage.
        sequence_logprobs = -sequence_cross_entropy

        if not self._average_logprobs_across_timesteps:
            # Sum of log-probabilities of every token (similar to what beam search outputs)
            # We need this other than `loss` to compute REINFORCE reward in question coding.
            sequence_logprobs *= program_lengths

        if not self._average_loss_across_timesteps:
            sequence_cross_entropy *= program_tokens_mask

        # "loss" is averaged sequence negative log-probability across sequences in the batch.
        output_dict = {
            "predicted_tokens": next_timestep,
            "loss": sequence_cross_entropy,
            "sequence_logprobs": sequence_logprobs
        }
        if not self.training:
            self._log2_perplexity(sequence_cross_entropy.mean().item())
        return output_dict

    def get_metrics(self) -> Dict[str, float]:
        """Return perplexity using the accumulated loss (mainly during validation)."""
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update({
                "perplexity": 2 ** self._log2_perplexity.get_metric(reset=True),
            })
        return all_metrics
