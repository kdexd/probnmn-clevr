import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicRNN(nn.Module):
    def __init__(self, rnn_model):
        super().__init__()
        self.rnn_model = rnn_model

    def forward(self, sequence_tokens, sequence_lengths):
        """A wrapper over pytorch's RNN to handle sequences of variable length.

        Parameters
        ----------
        sequence_tokens: torch.FloatTensor
            Input sequence tensor (padded).
            Shape: (batch, max_sequence_length, embed_size)
        sequence_lengths: torch.LongTensor
            Length of sequences (batch_size, )

        Returns
        -------
            Tensor of shape (batch_size, max_sequence_length, rnn_hidden_size) corresponding to
            the outputs of the RNN model at each time step of each input sequence.
        """
        sorted_len, sort_order, unsort_order = self._sort(sequence_lengths)
        sorted_seq_input = sequence_tokens.index_select(dim=0, index=sort_order)
        packed_seq_input = pack_padded_sequence(
            sorted_seq_input, lengths=sorted_len, batch_first=True)

        rnn_output, (h_n, c_n) = self.rnn_model(packed_seq_input)

        rnn_output_padded, _ = pad_packed_sequence(
            rnn_output, batch_first=True, total_length=sequence_tokens.size(1)
        )
        rnn_output_unsorted = rnn_output_padded.index_select(dim=0, index=unsort_order)
        return rnn_output_unsorted

    @staticmethod
    def _sort(lens):
        sorted_len, sort_order = torch.sort(lens.contiguous().view(-1), 0, descending=True)
        _, unsort_order = torch.sort(sort_order)
        sorted_len = list(sorted_len)
        return sorted_len, sort_order, unsort_order
