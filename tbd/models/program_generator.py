from typing import Optional

from allennlp.data import Vocabulary
from allennlp.models.encoder_decoders import SimpleSeq2Seq
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import sequence_cross_entropy_with_logits
import torch
from torch import nn


class ProgramGenerator(SimpleSeq2Seq):
    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 dropout: int = 0.0):
        # short-hand notations
        __question_vocab_size = vocabulary.get_vocab_size(namespace="questions")
        __program_vocab_size = vocabulary.get_vocab_size(namespace="programs")

        # source embedder converts tokenized source sequences to dense embeddings
        question_embedder = Embedding(__question_vocab_size, embedding_size, padding_index=0)
        question_embedder = BasicTextFieldEmbedder({"tokens": question_embedder})

        question_encoder = PytorchSeq2SeqWrapper(
            nn.LSTM(embedding_size, hidden_size, num_layers=2, dropout=dropout, batch_first=True)
        )
        super().__init__(
            vocab=vocabulary,
            source_embedder=question_embedder,
            encoder=question_encoder,
            max_decoding_steps=50,
            beam_size=1,
            target_namespace="programs",
        )

    def forward(self,
                question_tokens: torch.LongTensor,
                program_tokens: Optional[torch.LongTensor] = None,
                cross_entropy_reduction: Optional[str] = None):
        question_tokens = {"tokens": question_tokens}
        if program_tokens is not None:
            program_tokens = {"tokens": program_tokens}

        # keys: {"predictions", "loss"} if training
        output_dict = super().forward(question_tokens, program_tokens)

        # perform cross entropy reduction if specified
        if self.training and cross_entropy_reduction == "mean":
            output_dict["loss"] = output_dict["loss"].mean(dim=-1)
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
