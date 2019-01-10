from typing import Optional

from allennlp.data import Vocabulary
from allennlp.models.encoder_decoders import SimpleSeq2Seq
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
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
            nn.LSTM(embedding_size, hidden_size, dropout=dropout, batch_first=True)
        )
        super().__init__(
            vocab=vocabulary,
            source_embedder=question_embedder,
            encoder=question_encoder,
            max_decoding_steps=30,
            beam_size=3,
            target_namespace="programs",
        )

    def forward(self,
                question_tokens: torch.LongTensor,
                program_tokens: Optional[torch.LongTensor] = None):
        question_tokens = {"tokens": question_tokens}
        if program_tokens is not None:
            program_tokens = {"tokens": program_tokens}
        return super().forward(question_tokens, program_tokens)
