from typing import Dict, List, Optional, Tuple

from allennlp.data import Vocabulary
from allennlp.models.encoder_decoders import SimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import add_sentence_boundary_token_ids, sequence_cross_entropy_with_logits
from allennlp.training.metrics import Average
import torch
from torch import nn
from torch.nn import functional as F


class ProgramGenerator(SimpleSeq2Seq):
    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 dropout: float = 0.0,
                 beam_size: int = 1,
                 max_decoding_steps: int = 26):
        # Short-hand notations (source: questions, target: programs)
        __question_vocab_size = vocabulary.get_vocab_size(namespace="questions")
        __program_vocab_size = vocabulary.get_vocab_size(namespace="programs")

        # Source (question) embedder converts tokenized source sequences to dense embeddings.
        __question_embedder = Embedding(__question_vocab_size, embedding_size, padding_index=0)
        __question_embedder = BasicTextFieldEmbedder({"tokens": __question_embedder})

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        __encoder = PytorchSeq2SeqWrapper(
            nn.LSTM(embedding_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        )

        # attention mechanism between decoder context and encoder hidden states at each time step
        __attention = DotProductAttention()

        super().__init__(
            vocabulary,
            source_embedder=__question_embedder,
            encoder=__encoder,
            max_decoding_steps=max_decoding_steps,
            attention=__attention,
            beam_size=beam_size,
            target_namespace="programs",
        )

        # @@PADDING@@ and @@UNKNOWN@@ have same indices in all namespaces
        self._pad_index = vocabulary.get_token_index("@@PADDING@@", namespace="questions")
        self._unk_index = vocabulary.get_token_index("@@UNKNOWN@@", namespace="questions")

        # Record two metrics - BLEU score and perplexity.
        # super().__init__() already declared "self._bleu", perplexity = 2 ** average_val_loss.
        self._average_loss = Average()

    def forward(self,
                question_tokens: torch.LongTensor,
                program_tokens: Optional[torch.LongTensor] = None,
                sample_programs: bool = False) -> Dict[str, torch.Tensor]:
        """
        Override AllenNLP's forward, changing decoder logic. We concatenate encoder output to
        every time step during decoding, in contrast to just first time step.

        Extened Summary
        ---------------
        From AllenNLP documentation:

        Make forward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        question_tokens: torch.LongTensor
        program_tokens: torch.LongTensor, optional (default = None)
        sample_programs: bool, optional (default = None)
            Whether to perform categorical sampling to generate programs for question
            reconstrcutor. If False, then it will be treated as inference, and beam search
            will be performed. This would be ignored if program tokens are provided.

        Returns
        -------
        Dict[str, torch.Tensor]
        """

        # Add "@start@" and "@end@" tokens to program and question sequences.
        question_tokens, _ = add_sentence_boundary_token_ids(
            question_tokens, (question_tokens != self._pad_index), self._start_index, self._end_index
        )
        if program_tokens is not None:
            program_tokens, _ = add_sentence_boundary_token_ids(
                program_tokens, (program_tokens > 0), self._start_index, self._end_index
            )
        # Remove "@start@" from question anyway (t's being encoded).
        question_tokens = question_tokens[:, 1:]

        state = self._encode({"tokens": question_tokens})
        state = self._init_decoder_state(state)

        if program_tokens is not None:
            output_dict = self._forward_loop(state, {"tokens": program_tokens})

            if not self.training:
                # super class methods - left unchanged here
                state = self._init_decoder_state(state)
                output_dict.update(self._forward_beam_search(state))

                # BLEU score considers only the most likely beam.
                self._bleu(output_dict["predictions"][:, 0, :], program_tokens)
                self._average_loss(torch.mean(output_dict["loss"]).item())
        else:
            if sample_programs:
                # Sample programs for input to question reconstructor.
                # keys: {"predictions"}
                output_dict = self._sample(state)
            else:
                if not self.training:
                    # Purely inference - do beam search as usual
                    state = self._init_decoder_state(state)
                    output_dict = self._forward_beam_search(state)
                else:
                    output_dict = {}

        if not self.training:
            # Convert program sequences to string tokens.
            output_dict = self.decode(output_dict)

            if not sample_programs:
                # Keep best predictions (most likely beam) during inference
                output_dict["predictions"] = output_dict["predictions"][:, 0, :]
                output_dict["class_log_probabilities"] = output_dict["class_log_probabilities"][:, 0]
            else:
                # Trim output predictions to first "@end@" and pad the rest of sequence.
                predictions = output_dict["predictions"]
                trimmed_predictions = torch.zeros_like(output_dict["predictions"])
                for i, prediction in enumerate(predictions):
                    prediction_indices = list(prediction.detach().cpu().numpy())
                    if self._end_index in prediction_indices:
                        end_index = prediction_indices.index(self._end_index)
                        if end_index > 0:
                            trimmed_predictions[i][:end_index] = prediction[:end_index]                    
                    else:
                        trimmed_predictions[i] = prediction
                output_dict["predictions"] = trimmed_predictions

        return output_dict

    def _sample(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """During training, while sampling a sequence, sample from categorical distribution instead
        of performing greedy decoding.
        """

        # shape: (batch_size, max_input_sequence_length)
        source_mask = state["source_mask"]
        batch_size = source_mask.size()[0]

        # Initialize target predictions with the start index.
        # shape: (batch_size,)
        last_predictions = source_mask.new_full((batch_size,), fill_value=self._start_index)

        step_predictions: List[torch.Tensor] = []
        for timestep in range(self._max_decoding_steps):
            # shape: (batch_size,)
            input_choices = last_predictions
            # shape: (batch_size, num_classes)
            output_projections, state = self._prepare_output_projections(input_choices, state)
            # shape: (batch_size, num_classes)
            class_probabilities = F.softmax(output_projections, dim=-1)
            # Don't sample @@PADDING@@, @start@ and @@UNKNOWN@@ tokens
            class_probabilities[:, self._pad_index] = 0
            class_probabilities[:, self._start_index] = 0
            class_probabilities[:, self._unk_index] = 0
            # shape (predicted_classes): (batch_size, 1)
            predicted_classes = torch.multinomial(class_probabilities, 1)
            step_predictions.append(predicted_classes)
            # shape (predicted_classes): (batch_size,)
            last_predictions = predicted_classes.squeeze()

        # shape: (batch_size, num_decoding_steps)
        output_dict = {"predictions": torch.cat(step_predictions, 1)}
        return output_dict

    @staticmethod
    def _get_loss(logits: torch.LongTensor,
                  targets: torch.LongTensor,
                  target_mask: torch.LongTensor) -> torch.Tensor:
        """Override AllenNLP Seq2Seq model's provided `_get_loss` method, which returns sequence
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

    def get_metrics(self) -> Dict[str, float]:
        """Override AllenNLP's get_metric and return perplexity, along with BLEU."""
        all_metrics: Dict[str, float] = {}
        if not self.training:
            all_metrics.update(self._bleu.get_metric(reset=True))
            all_metrics.update({"perplexity": 2 ** self._average_loss.get_metric(reset=True)})
        return all_metrics
