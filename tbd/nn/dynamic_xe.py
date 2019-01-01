import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DynamicCrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(**kwargs)

    def forward(self, output, target, lengths):
        # remove padding from both sequences
        sorted_lengths, sort_order = torch.sort(lengths, dim=0, descending=True)

        sorted_output = output.index_select(dim=0, index=sort_order)
        sorted_target = target.index_select(dim=0, index=sort_order)

        # pack to prevent inclusion of padded tokens in cross-entropy calculation
        packed_output = pack_padded_sequence(
            sorted_output, sorted_lengths, batch_first=True
        )
        packed_target = pack_padded_sequence(
            sorted_target, sorted_lengths, batch_first=True
        )
        # access data of packed sequences, which collapses all padded tokens to nothing
        loss = self.criterion(packed_output.data, packed_target.data)
        return loss
