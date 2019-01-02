from torch import nn

from tbd.nn import DynamicRNN


class ProgramPriorVanillaLSTM(nn.Module):
    """
    A simpler class with Vanilla LSTM without any residual connections. This one is included
    to make sure that adding all the bells and whistles is not redundant.

    Refer docstrings and inline comments in ``ProgramPriorResidualLSTM``.
    """

    def __init__(self,
                 vocab_size: int = 44,
                 embedding_size: int = 128,
                 rnn_hidden_size: int = 256,
                 rnn_dropout: float = 0.25):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, embedding_size)
        self.decoder_lstm = nn.LSTM(
            embedding_size, rnn_hidden_size, num_layers=2,
            drouput=rnn_dropout, batch_first=True
        )
        self.decoder_linear = nn.Linear(rnn_hidden_size, vocab_size)
        self.decoder_lstm = DynamicRNN(self.decoder_lstm)

    def forward(self, program_tokens, program_lengths):
        program_embeds = self.embedder(program_tokens)
        decoder_output = self.decoder_lstm(program_embeds, program_lengths)
        decoder_logits = self.decoder_linear(decoder_output)
        return decoder_logits


class ProgramPriorResidualLSTM(nn.Module):
    """
    A language model which learns a prior over all the valid program sequences in CLEVR v1.0
    training split. It is a two-layered LSTM with a residual connection between the first
    and second layers.
    """

    def __init__(self,
                 vocab_size: int = 44,
                 embedding_size: int = 128,
                 rnn_hidden_size: int = 256,
                 rnn_dropout: float = 0.25):
        super().__init__()
        self.embedder = nn.Embedding(vocab_size, embedding_size)

        # declare two layers of single LSTM separately to support residual connection
        # dropout doesn't work directly on 1-layer LSTMs, can't be specified in LSTM constructor
        self.decoder_lstm_first = nn.LSTM(
            embedding_size, rnn_hidden_size, batch_first=True
        )
        # apply dropout before passing input to second layer (i.e. after residual connection)
        self.dropout = nn.Dropout(rnn_dropout)
        self.decoder_lstm_second = nn.LSTM(
            embedding_size, rnn_hidden_size, batch_first=True
        )

        # linear layer after first LSTM layer to project back to embedding dimension
        # necessary for residual connection, this adds to token embeddings
        self.decoder_linear_first = nn.Linear(rnn_hidden_size, embedding_size)

        # linear layer after second LSTM layer to get logits for possible tokens
        self.decoder_linear_second = nn.Linear(rnn_hidden_size, vocab_size)

        # wrap lstm with DynamicRNN module to handle programs of variable length
        self.decoder_lstm_first = DynamicRNN(self.decoder_lstm_first)
        self.decoder_lstm_second = DynamicRNN(self.decoder_lstm_second)

    def forward(self, program_tokens, program_lengths):
        """
        Given a batch of tokenized programs (padded), predict the logits of next time-step.

        Parameters
        ----------
        program_tokens: torch.FloatTensor
            Batch of program tokens (padded), shape: (batch, max_program_length).
        program_lengths: torch.LongTensor
            Corresponding lengths of (unpadded) programs, shape: (batch, ).

        Returns
        -------
        torch.FloatTensor
            Logits from decoder, shape: (batch, max_program_length, vocab_size).
        """

        # shape: (batch, max_program_length, embedding_dim)
        program_embeds = self.embedder(program_tokens)

        # shape: (batch, max_program_length, rnn_hidden_dim)
        decoder_output_first = self.decoder_lstm_first(program_embeds, program_lengths)

        # shape: (batch, max_program_length, embedding_size)
        linear_output_first = self.decoder_linear_first(decoder_output_first)
        residual_output = program_embeds + linear_output_first
        residual_output = self.dropout(residual_output)

        # shape: (batch, max_program_length, rnn_hidden_dim)
        decoder_output_second = self.decoder_lstm_second(residual_output, program_lengths)

        # predicts the sequence at next time-step
        # shape: (batch, max_program_length, vocab_size)
        decoder_logits = self.decoder_linear_second(decoder_output_second)

        # cross entropy loss will be between:
        # program_tokens[:, 1:] and decoder_logits[:, :-1]
        return decoder_logits


# default program prior model used in all experiments
ProgramPrior = ProgramPriorResidualLSTM
