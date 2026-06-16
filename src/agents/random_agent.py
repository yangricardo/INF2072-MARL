"""Baseline aleatório.

Extraído de: Código/Executa Experimento Randomico.ipynb (classe RandomAgent).

Escolhe ações uniformemente ao acaso e não aprende. Mantém a mesma interface do
``IDQNAgent`` (incluindo ``remember``/``optimize``/``*_checkpoint`` como no-ops)
para ser usado diretamente no mesmo loop de treino em ``training.py``.
"""

import random


class RandomAgent:
    def __init__(self, state_dim, action_dim, agent_id, config=None):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.steps_done = 0
        self.total_episodes = 0
        self.memory = []  # B9: compatibilidade com _should_optimize()

    def select_action(self, state, training=True):
        self.steps_done += 1
        return random.randrange(self.action_dim)

    def get_epsilon(self):
        # Sempre "explorando" — ações 100% aleatórias.
        return 1.0

    # --- No-ops: o baseline não tem memória nem aprendizado --------------
    def remember(self, state, action, reward, next_state, done):
        pass

    def optimize(self):
        return 0

    def save_checkpoint(self, save_dir):
        return False

    def load_checkpoint(self, load_dir, agent_id):
        return False
