from typing import Dict, Optional, Tuple

from allennlp.data import Vocabulary
from allennlp.models.encoder_decoders import SimpleSeq2Seq
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.beam_search import BeamSearch
from allennlp.nn.util import sequence_cross_entropy_with_logits, get_final_encoder_states
from allennlp.training.metrics import BLEU
import torch
from torch import nn


class ProgramGenerator(SimpleSeq2Seq):
    def __init__(self,
                 vocabulary: Vocabulary,
                 embedding_size: int = 128,
                 hidden_size: int = 64,
                 dropout: float = 0.0,
                 beam_size: int = 1,
                 max_decoding_steps: int = 30):
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
        super().__init__(
            vocabulary,
            source_embedder=__question_embedder,
            encoder=__encoder,
            max_decoding_steps=max_decoding_steps,
            attention=None,
            beam_size=beam_size,
            target_namespace="programs",
        )
        # @@@@@@@@@@ GLARINGLY VISIBLE NOTE @@@@@@@@@@
        # THIS LINE IS DIFFERENT FROM SUPER CLASS OF ALLENNLP

        # If using attention, a weighted average over encoder outputs will be concatenated
        # to the previous target embedding to form the input to the decoder at each
        # time step.

        # Otherwise, final encoder output will be concatenated to the previous target embedding
        # to form decoder input at each time step
        self._decoder_input_dim = self._decoder_output_dim + embedding_size
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # We'll use an LSTM cell as the recurrent cell that produces a hidden state
        # for the decoder at each time step.
        self._decoder_cell = nn.LSTMCell(self._decoder_input_dim, self._decoder_output_dim)

        # We project the hidden state from the decoder into the output vocabulary space
        # in order to get log probabilities of each target token, at each time step.
        self._output_projection_layer = nn.Linear(self._decoder_output_dim, __program_vocab_size)

    def forward(self,
                question_tokens: torch.LongTensor,
                program_tokens: Optional[torch.LongTensor] = None) -> Dict[str, torch.Tensor]:
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

        Returns
        -------
        Dict[str, torch.Tensor]
        """
        state = self._encode({"tokens": question_tokens})

        if program_tokens is not None:
            program_tokens = {"tokens": program_tokens}

        state = self._init_decoder_state(state)
        # The `_forward_loop` decodes the input sequence and computes the loss during training
        # and validation.
        output_dict = self._forward_loop(state, program_tokens)

        if not self.training:
            state = self._init_decoder_state(state)
            predictions = self._forward_beam_search(state)
            output_dict.update(predictions)
            if program_tokens and self._bleu:
                # shape: (batch_size, beam_size, max_sequence_length)
                top_k_predictions = output_dict["predictions"]
                # shape: (batch_size, max_predicted_sequence_length)
                best_predictions = top_k_predictions[:, 0, :]
                self._bleu(best_predictions, program_tokens["tokens"])

        return output_dict

    def _prepare_output_projections(self,
                                    last_predictions: torch.Tensor,
                                    state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:  # pylint: disable=line-too-long
        """Override AllenNLP's _prepare_output_projection to concatenate encoder
        outputs to each decoding time step.

        Extended Summary
        ----------------
        From AllenNLP's documentation:

        Decode current state and last prediction to produce produce projections
        into the target space, which can then be used to get probabilities of
        each target token for the next step.
        Inputs are the same as for `take_step()`.
        """
        # shape: (group_size, max_input_sequence_length, encoder_output_dim)
        encoder_outputs = state["encoder_outputs"]

        encoder_final_outputs = get_final_encoder_states(
            state["encoder_outputs"],
            state["source_mask"],
            self._encoder.is_bidirectional()
        )

        # shape: (group_size, max_input_sequence_length)
        source_mask = state["source_mask"]

        # shape: (group_size, decoder_output_dim)
        decoder_hidden = state["decoder_hidden"]

        # shape: (group_size, decoder_output_dim)
        decoder_context = state["decoder_context"]

        # shape: (group_size, target_embedding_dim)
        embedded_input = self._target_embedder(last_predictions)

        if self._attention:
            # shape: (group_size, encoder_output_dim)
            attended_input = self._prepare_attended_input(decoder_hidden, encoder_outputs, source_mask)

            # shape: (group_size, decoder_output_dim + target_embedding_dim)
            decoder_input = torch.cat((attended_input, embedded_input), -1)
        else:
            # shape: (group_size, encoder_output_dm + target_embedding_dim)
            decoder_input = torch.cat((encoder_final_outputs, embedded_input), -1)

        # shape (decoder_hidden): (batch_size, decoder_output_dim)
        # shape (decoder_context): (batch_size, decoder_output_dim)
        decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input,
                (decoder_hidden, decoder_context))

        state["decoder_hidden"] = decoder_hidden
        state["decoder_context"] = decoder_context

        # shape: (group_size, num_classes)
        output_projections = self._output_projection_layer(decoder_hidden)

        return output_projections, state

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
