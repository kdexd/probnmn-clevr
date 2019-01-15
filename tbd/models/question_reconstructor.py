from typing import Dict, Optional, Tuple

from allennlp.data import Vocabulary
from allennlp.models.encoder_decoders import SimpleSeq2Seq
from allennlp.modules.attention import DotProductAttention
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import Average
import torch
from torch import nn


class QuestionReconstructor(SimpleSeq2Seq):
    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 dropout: float = 0.0,
                 beam_size: int = 1,
                 max_decoding_steps: int = 45):
        # Short-hand notations (source: programs, target: questions)
        __program_vocab_size = vocabulary.get_vocab_size(namespace="programs")
        __question_vocab_size = vocabulary.get_vocab_size(namespace="questions")

        # Source (question) embedder converts tokenized source sequences to dense embeddings.
        __program_embedder = Embedding(__program_vocab_size, embedding_size, padding_index=0)
        __program_embedder = BasicTextFieldEmbedder({"tokens": __program_embedder})

        # Encodes the sequence of source embeddings into a sequence of hidden states.
        __encoder = PytorchSeq2SeqWrapper(
            nn.LSTM(embedding_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        )

        # attention mechanism between decoder context and encoder hidden states at each time step
        __attention = DotProductAttention()

        super().__init__(
            vocabulary,
            source_embedder=__program_embedder,
            encoder=__encoder,
            max_decoding_steps=max_decoding_steps,
            attention=__attention,
            beam_size=beam_size,
            target_namespace="questions",
        )

        # Record two metrics - BLEU score and perplexity.
        # super().__init__() already declared "self._bleu", perplexity = 2 ** average_val_loss.
        self._average_loss = Average()

    def forward(self,
                program_tokens: torch.LongTensor,
                question_tokens: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
        """
        Override AllenNLP's forward, changing decoder logic. We concatenate encoder output to
        every time step during decoding, in contrast to just first time step.

        Extened Summary
        ---------------
        From AllenNLP documentation:

        Make forward pass with decoder logic for producing the entire target sequence.

        Parameters
        ----------
        program_tokens: torch.LongTensor
        question_tokens: torch.LongTensor, optional (default = None)

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode({"tokens": program_tokens})
        state = self._init_decoder_state(state)

        if question_tokens is not None:
            question_tokens = {"tokens": question_tokens}

        # keys: {"predictions", "loss"}, question_tokens will never be None.
        output_dict = self._forward_loop(state, question_tokens)
        if not self.training:
            # keys: {"predictions", "class_log_probabilities"}
            output_dict.update(self._forward_beam_search(state))
            # Keep only best prediction (most likely beam).
            output_dict["predictions"] = output_dict["predictions"][:, 0, :]
            output_dict["class_log_probabilities"] = output_dict["class_log_probabilities"][:, 0]

            # Convert predictions to string tokens.
            output_dict = self.decode(output_dict)

            # Record metrics while performing validation.
            if question_tokens:
                self._bleu(output_dict["predictions"], question_tokens["tokens"])
                self._average_loss(torch.mean(output_dict["loss"]).item())

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
