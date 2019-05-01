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
                    "predictions": torch.Tensor (shape: (batch_size, max_sequence_length - 1)),
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

    def sample(
        self, num_samples: int = 1, max_sequence_length: int = 28
    ) -> Dict[str, torch.Tensor]:
        r"""
        Using @start@ token at first time-step, perform categorical sampling and sample program
        sequences freely, all sequences would be padded after encountering first @end@ token.

        This method is mainly useful in checking coherence and sensitivity of our model's beliefs.

        Parameters
        ----------
        num_samples: int, optional (default = 1)
            Number of program_samples to generate.
        max_sequence_length: int, optional (default = 28)
            Maximum decoding steps while sampling programs. This includes @start@ token. Output
            sequences will be one time-step smaller, excluding @start@.

        Returns
        -------
        Dict[str, torch.Tensor]
            A dict with predictions and sequence log-probabilities (averaged across time-steps).
            This would acutally return negative log-probabilities and name it "loss" for API
            consistency. The dict structure looks like::

                {
                    "predictions": torch.Tensor (shape: (batch_size, max_sequence_length - 1)),
                    "loss": torch.Tensor (shape: (batch_size, ))
                }
        """

        # Get device of any member in this module to initialize fresh tensors on that device.
        device = self._output_layer.weight.device

        # Take out the PyTorch module from AllenNLP's wrapper.
        encoder = self._encoder._module

        # Create a tensor of @start@ tokens on same device as this module.
        # shape: (num_samples, )
        last_predictions = torch.full(
            (num_samples, 1), fill_value=self._start_index, device=device
        ).long()

        # Initialize hidden and cell states as zeroes.
        # shape: (num_layers, num_samples, hidden_size)
        hidden_state = torch.zeros(
            (encoder.num_layers, num_samples, encoder.hidden_size), device=device
        )
        cell_state = torch.zeros(
            (encoder.num_layers, num_samples, encoder.hidden_size), device=device
        )

        step_logits: List[torch.Tensor] = []
        step_logprobs: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        for timestep in range(max_sequence_length - 1):
            # shape: (num_samples, 1)
            input_choices = last_predictions

            # shape: (num_samples, 1, input_size)
            embedded_tokens = self._embedder({"programs": last_predictions})

            # shape: (num_samples, 1, hidden_size)
            encoded_tokens, (hidden_state, cell_state) = encoder(
                embedded_tokens, (hidden_state, cell_state)
            )

            # shape: (num_samples, 1, input_size)
            output_projection = self._projection_layer(encoded_tokens)

            # shape: (num_samples, 1, vocab_size)
            output_logits = self._output_layer(output_projection)

            output_class_probabilities = F.softmax(output_logits, dim=-1)
            output_class_logprobs = F.log_softmax(output_projection, dim=-1)

            # Perform categorical sampling, don't sample @@PADDING@@, @@UNKNOWN@@, @start@.
            output_class_probabilities[:, :, self._start_index] = 0
            output_class_probabilities[:, :, self._pad_index] = 0
            output_class_probabilities[:, :, self._unk_index] = 0

            # shape: (num_samples, 1)
            predicted_classes = torch.multinomial(output_class_probabilities.squeeze(1), 1)

            # Pick predictions and their logprobs and append to lists.
            last_predictions = predicted_classes
            output_class_logprobs = torch.gather(
                output_class_logprobs, 2, predicted_classes.unsqueeze(1)
            )

            # List of tensors, shape: (num_samples, 1, vocab_size)
            step_predictions.append(last_predictions)
            step_logits.append(output_projection)
            step_logprobs.append(output_class_logprobs.squeeze(-1))

        # shape: (batch_size, max_sequence_length)
        predictions = torch.cat(step_predictions, 1)

        # Trim predictions after first "@end@" token.
        # shape: (batch_size, num_decoding_steps)
        trimmed_predictions = torch.zeros_like(predictions)
        for i, prediction in enumerate(predictions):
            prediction_indices = list(prediction.detach().cpu().numpy())
            if self._end_index in prediction_indices:
                end_index = prediction_indices.index(self._end_index)
                if end_index > 0:
                    trimmed_predictions[i][: end_index + 1] = prediction[: end_index + 1]
            else:
                trimmed_predictions[i] = prediction
        predictions = trimmed_predictions

        # Log-probabilities at each time-step (without teacher forcing).
        logprobs = torch.cat(step_logprobs, 1)
        prediction_mask = (predictions != self._pad_index).float()

        # Average the sequence logprob across time-steps. This ensures equal importance of all
        # sequences irrespective of their lengths.
        sequence_logprobs = logprobs.mul(prediction_mask).sum(-1)
        prediction_lengths = prediction_mask.sum(-1)
        # shape: (batch_size, )
        sequence_logprobs /= prediction_lengths + 1e-12

        # Sort predictions and logprobs in ascending order. Most probble sequence has the lowest
        # "loss" (negative logprobs).
        sequence_logprobs_sorting = (- sequence_logprobs).sort()[1]
        predictions = predictions[sequence_logprobs_sorting]
        sequence_logprobs = sequence_logprobs[sequence_logprobs_sorting]

        output_dict = {"predictions": predictions, "loss": - sequence_logprobs}
        return output_dict
