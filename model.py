import torch
from torch import nn

class mlp(nn.Module):

    def __init__(self, in_width, out_width, hidden_width, depth=3):
        super().__init__()

        hidden_layers = []
        for _ in range(depth):
            hidden_layers.append(nn.Linear(hidden_width, hidden_width))
            hidden_layers.append(nn.ReLU())

        self.mlp = nn.Sequential(
            nn.Linear(in_width, hidden_width),
            *hidden_layers,
            nn.Linear(hidden_width, out_width)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits

class mlp_dropout(nn.Module):

    def __init__(self, in_width, out_width, hidden_width, depth=3, dropout=0.1):
        super().__init__()

        hidden_layers = []
        for _ in range(depth):
            hidden_layers.append(nn.Linear(hidden_width, hidden_width))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(p=dropout))

        self.mlp = nn.Sequential(
            nn.Linear(in_width, hidden_width),
            *hidden_layers,
            nn.Linear(hidden_width, out_width)
        )

    def forward(self, x):
        logits = self.mlp(x)
        return logits
