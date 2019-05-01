from typing import Dict, List, Optional

from allennlp.data import Vocabulary
from allennlp.models.encoder_decoders import SimpleSeq2Seq as AllenNlpSimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Average, SequenceAccuracy, UnigramRecall
import torch
from torch import nn
from torch.nn import functional as F


class Seq2SeqBase(AllenNlpSimpleSeq2Seq):
    r"""
    A wrapper over AllenNLP's SimpleSeq2Seq class. This serves as a base class for the
    :class:`~probnmn.models.program_generator.ProgramGenerator` and
    :class:`~probnmn.models.question_reconstructor.QuestionReconstructor`. The key differences
    from super class are:

        1. This class doesn't use beam search, it performs categorical sampling or greedy decoding
           as explicitly passed on :meth:`forward` call.
        2. This class records four metrics: perplexity, sequence_accuracy, word error rate and
           BLEU score.
        3. Has sensible defaults for super class (dot-product attention, embedding etc.).

    Parameters
    ----------
    vocabulary: allennlp.data.vocabulary.Vocabulary
        AllenNLP's vocabulary. This vocabulary has three namespaces - "questions", "programs" and
        "answers", which contain respective token to integer mappings.
    source_namespace: str, required
        Namespace for source tokens,
        "programs" for :class:`~probnmn.models.question_reconstructor.QuestionReconstructor` and
        "questions" for :class:`~probnmn.models.program_generator.ProgramGenerator`.
    target_namespace: str, required
        Namespace for target tokens, "programs" for ProgramGenerator and "questions" for
        QuestionReconstructor.
    input_size : int, optional (default = 256)
        The dimension of the inputs to the LSTM.
    hidden_size : int, optional (default = 256)
        The dimension of the outputs of the LSTM.
    num_layers: int, optional (default = 2)
        Number of recurrent layers of the LSTM.
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        source_namespace: str,
        target_namespace: str,
        input_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        max_decoding_steps: int = 30,
    ):

        # @@PADDING@@, @@UNKNOWN@@, @start@, @end@ have same indices in all namespaces.
        self._pad_index = vocabulary.get_token_index("@@PADDING@@", namespace=source_namespace)
        self._unk_index = vocabulary.get_token_index("@@UNKNOWN@@", namespace=source_namespace)
        self._end_index = vocabulary.get_token_index("@end@", namespace=source_namespace)
        self._start_index = vocabulary.get_token_index("@start@", namespace=source_namespace)

        # Short-hand notations.
        __source_vocab_size = vocabulary.get_vocab_size(namespace=source_namespace)
        __target_vocab_size = vocabulary.get_vocab_size(namespace=target_namespace)

        # Source embedder converts tokenized source sequences to dense embeddings.
        __source_embedder = BasicTextFieldEmbedder(
            {"tokens": Embedding(__source_vocab_size, input_size, padding_index=self._pad_index)}
        )

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        __encoder = PytorchSeq2SeqWrapper(
            nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        )

        # Attention mechanism between decoder context and encoder hidden states at each time step.
        __attention = DotProductAttention()

        super().__init__(
            vocabulary,
            source_embedder=__source_embedder,
            encoder=__encoder,
            max_decoding_steps=max_decoding_steps,
            attention=__attention,
            target_namespace=target_namespace,
            use_bleu=True,
        )
        # Record four metrics - perplexity, sequence accuracy, word error rate and BLEU score.
        # super().__init__() already declared "self._bleu",
        # perplexity = 2 ** average_val_loss
        # word error rate = 1 - unigram recall
        self._log2_perplexity = Average()
        self._sequence_accuracy = SequenceAccuracy()
        self._unigram_recall = UnigramRecall()

    def forward(
        self,
        source_tokens: torch.LongTensor,
        target_tokens: Optional[torch.LongTensor] = None,
        decoding_strategy: str = "sampling",
    ) -> Dict[str, torch.Tensor]:

        r"""
        Override AllenNLP's forward, changing decoder logic. Perform either categorical sampling
        or greedy decoding as per specified.

        Parameters
        ----------
        source_tokens: torch.LongTensor
            Tokenized source sequences padded to maximum length. These are not padded with
            @start@ and @end@ sentence boundaries. Shape: (batch_size, max_source_length)
        target_tokens: torch.LongTensor, optional (default = None)
            Tokenized target sequences padded to maximum length. These are not padded with
            @start@ and @end@ sentence boundaries. Shape: (batch_size, max_target_length)
        decoding_strategy: str, optional (default = "sampling")
            How to perform decoding? One of "sampling" or "greedy".

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        # Add "@start@" and "@end@" tokens to source and target sequences.
        source_tokens, _ = add_sentence_boundary_token_ids(
            source_tokens, (source_tokens != self._pad_index), self._start_index, self._end_index
        )
        if target_tokens is not None:
            target_tokens, _ = add_sentence_boundary_token_ids(
                target_tokens,
                (target_tokens != self._pad_index),
                self._start_index,
                self._end_index,
            )
        # Remove "@start@" from source sequences anyway (it's being encoded).
        source_tokens = {"tokens": source_tokens[:, 1:]}
        if target_tokens is not None:
            target_tokens = {"tokens": target_tokens}

        # _encode and _init_decoder_state are super class methods, left untouched.
        # keys: {"encoder_outputs", "source_mask"}
        state = self._encode(source_tokens)

        # keys: {"encoder_outputs", "source_mask", "decoder_hidden", "decoder_context"}
        state = self._init_decoder_state(state)

        # The `_forward_loop` decodes the input sequence and computes the loss during training
        # and validation.
        # keys: {"predictions", "loss"}
        output_dict = self._forward_loop(state, target_tokens, decoding_strategy)

        return output_dict

    def _forward_loop(
        self,
        state: Dict[str, torch.FloatTensor],
        target_tokens: Dict[str, torch.LongTensor] = None,
        decoding_strategy: str = "sampling",
    ) -> Dict[str, torch.Tensor]:

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size()[0]

        if target_tokens:
            # shape: (batch_size, max_target_sequence_length)
            targets = target_tokens["tokens"]
            _, target_sequence_length = targets.size()

            # The last input from the question is either padding or the end symbol.
            # Either way, we don't have to process it.
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        # Initialize question predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_logits: List[torch.Tensor] = []
        step_logprobs: List[torch.Tensor] = []
        step_predictions: List[torch.Tensor] = []

        for timestep in range(num_decoding_steps):
            if self.training and torch.rand(1).item() < self._scheduled_sampling_ratio:
                # Use gold tokens at test time and at a rate of 1 - _scheduled_sampling_ratio
                # during training.
                # shape: (batch_size,)
                input_choices = last_predictions
            elif not target_tokens:
                # shape: (batch_size,)
                input_choices = last_predictions
            else:
                # shape: (batch_size,)
                input_choices = targets[:, timestep]

            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)
            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)
            class_logprobs = F.log_softmax(output_projections, dim=-1)

            # NOTE -------------------------------------------------------------------------------
            # This differs from super()._forward_loop(...)
            if decoding_strategy == "greedy":
                _, predicted_classes = torch.max(class_probabilities, 1)
            elif decoding_strategy == "sampling":
                # Perform categorical sampling, don't sample @@PADDING@@, @@UNKNOWN@@, @start@.
                class_probabilities[:, self._pad_index] = 0
                class_probabilities[:, self._unk_index] = 0
                class_probabilities[:, self._start_index] = 0
                predicted_classes = torch.multinomial(class_probabilities, 1).squeeze()
            # ------------------------------------------------------------------------------------

            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes
            class_logprobs = class_logprobs[torch.arange(batch_size), predicted_classes]

            # List of tensors, shape: (batch_size, 1, num_classes)
            step_predictions.append(last_predictions.unsqueeze(1))
            step_logits.append(output_projections.unsqueeze(1))
            step_logprobs.append(class_logprobs.unsqueeze(1))

        # shape: (batch_size, num_decoding_steps)
        predictions = torch.cat(step_predictions, 1)
        # Trim predictions after first "@end@" token.
        predictions = self._trim_predictions(predictions)

        # Log-probabilities at each time-step (without teacher forcing).
        # This will be `loss`, to compute REINFORCE reward in question coding.
        # In presence of target tokens, we replace this as cross entropy loss (teacher forcing).
        logprobs = torch.cat(step_logprobs, 1)
        prediction_mask = (predictions != self._pad_index).float()

        # Average the sequence logprob across time-steps. Hence our REINFORCE reward will be
        # length-normalized sequence log-probability. This ensures equal importance of all
        # sequences irrespective of their lengths.
        sequence_logprobs = logprobs.mul(prediction_mask).sum(-1)
        prediction_lengths = prediction_mask.sum(-1)
        # shape: (batch_size, )
        sequence_logprobs /= prediction_lengths + 1e-12

        output_dict = {"predictions": predictions, "loss": -sequence_logprobs}
        if target_tokens:
            # shape: (batch_size, num_decoding_steps, num_classes)
            logits = torch.cat(step_logits, 1)
            # shape: (batch_size, max_sequence_length)
            target_mask = targets != self._pad_index
            # shape: (batch_size, )
            sequence_cross_entropy = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = sequence_cross_entropy

            relevant_targets = targets[:, 1:]

            if not self.training:
                # Record BLEU, perplexity, word error rate and sequence accuracy during validation.
                self._bleu(predictions, targets)
                self._log2_perplexity(sequence_cross_entropy.mean().item())

                # Sequence accuracy and unigram recall expect a beam dimension.
                # Compare generated sequences without "@start@" token.
                self._sequence_accuracy(
                    predictions[:, : relevant_targets.size(-1)].unsqueeze(1),
                    relevant_targets,
                    (relevant_targets != self._pad_index).long(),
                )
                self._unigram_recall(
                    predictions[:, : relevant_targets.size(-1)].unsqueeze(1),
                    relevant_targets,
                    (relevant_targets != self._pad_index).long(),
                )

        return output_dict

    def _trim_predictions(self, predictions: torch.LongTensor):
        r"""
        Trim output predictions at first "@end@" and pad the rest of sequence.
        This includes "@end@" as last token in trimmed sequence.
        """
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
        return trimmed_predictions

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor
    ):
        r"""
        Override AllenNLP Seq2Seq model's provided ``_get_loss`` method, which returns sequence
        cross entropy averaged over batch by default. Instead, provide sequence cross entropy of
        each sequence in a batch separately.

        Extended Summary
        ----------------
        From AllenNLP documentation:

        Compute loss.
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes
        cross entropy loss while taking the mask into account.
        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.
        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        # shape: (batch_size, num_decoding_steps)
        relevant_targets = targets[:, 1:].contiguous()

        # shape: (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()

        return sequence_cross_entropy_with_logits(
            logits, relevant_targets, relevant_mask, average=None
        )

    def get_metrics(self, reset: bool = True) -> Dict[str, float]:
        r"""
        Return recorded metrics - perplexity, sequence accuracy, word error rate, BLEU.

        Parameters
        ----------
        reset: bool, optional (default = True)
            Whether to reset the accumulated metrics after retrieving them.

        Returns
        -------
        Dict[str, float]
            A dictionary with metrics::

                {
                    "perplexity",
                    "sequence_accuracy",
                    "word_error_rate",
                    "BLEU"
                }
        """

        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._bleu.get_metric(reset=True))
            all_metrics.update(
                {
                    "perplexity": 2 ** self._log2_perplexity.get_metric(reset=reset),
                    "sequence_accuracy": self._sequence_accuracy.get_metric(reset=reset),
                    "word_error_rate": 1 - self._unigram_recall.get_metric(reset=reset),
                }
            )
        return all_metrics
