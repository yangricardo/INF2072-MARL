"""Redes neurais usadas pelos agentes.

Reúne as arquiteturas extraídas dos scripts/notebooks originais (proveniência
indicada em cada classe).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class AgentNet(nn.Module):
    """Rede Q individual por agente (VDN), com LayerNorm e init ortogonal.

    Extraído de: Código/Executa Experimento VDN.ipynb (classe AgentNet).
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=2 ** 0.5)
                nn.init.constant_(m.bias, 0.0)
        # última camada com gain menor → outputs inicialmente próximos de 0
        last_layer = self.net[-1]
        if isinstance(last_layer, nn.Linear):
            nn.init.orthogonal_(last_layer.weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QMixer(nn.Module):
    """Mixer monotônico do QMIX (hiper-redes condicionadas no estado global).

    Extraído de: Código/Executa - Experimento.ipynb (classe QMixer).
    """

    def __init__(self, n_agents, state_dim, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * n_agents),
        )
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, agent_qs, states):
        batch_size = agent_qs.size(0)

        # Ensure non-negativity of weights (monotonicity constraint for QMIX)
        w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.hidden_dim, self.n_agents)
        b1 = self.hyper_b1(states).view(batch_size, self.hidden_dim, 1)

        hidden = torch.bmm(w1, agent_qs.unsqueeze(2)) + b1
        hidden = torch.relu(hidden)

        w2 = torch.abs(self.hyper_w2(states)).view(batch_size, 1, self.hidden_dim)
        b2 = self.hyper_b2(states).view(batch_size, 1, 1)

        q_total = torch.bmm(w2, hidden) + b2
        return q_total.squeeze(-1)


class ActorNetwork(nn.Module):
    """Rede de política (ator) por agente — MAPPO/HATRPO.

    Extraído de: Código/Executa Experimento MPPO.ipynb (classe ActorNetwork).
    """

    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1), logits


class CriticNetwork(nn.Module):
    """Rede de valor (crítico) centralizado — MAPPO.

    Extraído de: Código/Executa Experimento MPPO.ipynb (classe CriticNetwork).
    """

    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.network(x)


class ImprovedActorNetwork(nn.Module):
    """Ator com camadas residuais + LayerNorm — HATRPO.

    Extraído de: Código/Executa Experimento HATRPO.ipynb (classe ImprovedActorNetwork).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, action_dim)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state):
        x = self.activation(self.input_layer(state))
        x = self.layer_norms[0](x)
        x = self.dropout(x)
        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = self.activation(layer(x))
            x = self.layer_norms[i + 1](x)
            x = self.dropout(x)
            x = x + residual
        logits = self.output_layer(x)
        return F.softmax(logits, dim=-1), logits


class ImprovedCriticNetwork(nn.Module):
    """Crítico centralizado profundo — HATRPO.

    Extraído de: Código/Executa Experimento HATRPO.ipynb (classe ImprovedCriticNetwork).
    """

    def __init__(self, global_state_dim, hidden_dim=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.input_layer = nn.Linear(global_state_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 1)]
        )
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, global_state):
        x = self.activation(self.input_layer(global_state))
        x = self.layer_norms[0](x)
        x = self.dropout(x)
        for i, layer in enumerate(self.hidden_layers):
            residual = x
            x = self.activation(layer(x))
            x = self.layer_norms[i + 1](x)
            x = self.dropout(x)
            x = x + residual  # residual connection (matching ImprovedActorNetwork)
        return self.output_layer(x)
