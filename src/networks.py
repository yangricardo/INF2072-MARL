"""Redes neurais usadas pelos agentes."""

import torch.nn as nn


class ImprovedDQN(nn.Module):
    """MLP de 3 camadas ocultas com dropout e inicialização Xavier."""

    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.2):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        return self.network(x)
