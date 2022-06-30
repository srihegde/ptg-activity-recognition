import pdb

import torch
from torch import nn


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
        batch_size = x[0].shape[0]
        # pdb.set_trace()
        x = torch.stack(x).squeeze()
        x = self.fc1(x)
        x, _ = self.lstm1(x)

        x = x.contiguous().view(-1, self.n_hidden)
        x = self.fc2(x)
        out = x.view(batch_size, -1, self.act_classes)[:, -1, :]
        pred = torch.argmax(out, dim=1)
        loss = self.loss(out, data[1])

        return out, pred, loss

    def init_hidden(self, batch_size):
        """Initializes hidden state."""
        # Create a new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if train_on_gpu:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda()
        else:
            hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_()

        return hidden
