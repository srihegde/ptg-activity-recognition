import pdb

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TemporalModule(nn.Module):
    """docstring for TemporalModule."""

    def __init__(
        self, act_classes: int, n_hidden: int = 128, n_layers: int = 1, drop_prob: float = 0.5
    ):
        super(TemporalModule, self).__init__()
        self.act_classes = act_classes
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.fc1 = nn.Linear(2048, n_hidden)
        self.fc2 = nn.Linear(n_hidden, act_classes)
        self.lstm1 = nn.LSTM(n_hidden, n_hidden, n_layers)
        self.dropout = nn.Dropout(drop_prob)

        # Loss functions
        self.loss = nn.CrossEntropyLoss()

    def forward(self, data):
        x = data[0]["feats"]
        batch_size = x.shape[1]
        # x = torch.stack(x).squeeze(2)
        x = self.fc1(x)
        # pdb.set_trace()
        packed_input = pack_padded_sequence(
            x, data[1].tolist(), batch_first=False, enforce_sorted=False
        )
        packed_output, _ = self.lstm1(packed_input)
        x, _ = pad_packed_sequence(packed_output, batch_first=True)

        x = x.contiguous().view(-1, self.n_hidden)
        x = self.fc2(x)
        out = x.view(batch_size, -1, self.act_classes)[:, -1, :]
        pred = torch.argmax(out, dim=1)
        loss = self.loss(out, data[0]["act"].long())

        return out, pred, loss
