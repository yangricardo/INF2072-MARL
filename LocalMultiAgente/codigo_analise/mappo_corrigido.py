#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAPPO CORRIGIDO (standalone, roda local) — INF2072 / PUC-DI.

Este script e uma copia FIEL do notebook `Executa Experimento MPPO.ipynb`
(mesmo ambiente, mesmas redes, mesmos hiperparametros) com UMA correcao:
o metodo MAPPOAgent.select_action passou a AMOSTRAR da distribuicao da
politica (Categorical) e registrar o log_prob correto, em vez de usar
epsilon-greedy + argmax — que invalidava a razao de importancia do PPO e
fazia o MAPPO ficar pior que o aleatorio.

Objetivo: medir se, com o bug corrigido, o MAPPO passa a aprender.

Uso:
    python mappo_corrigido.py --episodes 30      # teste de cronometragem
    python mappo_corrigido.py --episodes 1500    # treino de verdade

Saida: pasta `mappo_fixed_results_<timestamp>/` com CSV + graficos + relatorio,
no mesmo formato dos outros experimentos (entra direto no comparar_metricas.py).
"""
import argparse
import os as _os


def _pick_device():
    import torch
    forced = _os.environ.get("MAPPO_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


"""
MAPPO (Multi-Agent Proximal Policy Optimization) para Warehouse
- 3000 episódios de treinamento
- Parâmetros otimizados para estabilidade
- Gráficos sem média móvel
- Barreiras Y removidas aleatoriamente
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import warnings
import time
import gc

warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO DO AMBIENTE ====================
MAP_CONFIG = {
    'height': 12,
    'width': 8,
    'grid': [
        ['R1', '0', '0', '0', '0', '0', '0', 'R2'],
        ['0', 'Y', 'A', 'A', 'A', 'A', '0', '0'],
        ['0', '0', 'A', 'A', 'A', 'A', '0', '0'],
        ['X', '0', '0', '0', 'X', '0', 'Y', '0'],
        ['0', 'Y', 'X', '0', '0', '0', '0', '0'],
        ['0', '0', '0', 'X', '0', 'Y', 'X', 'X'],
        ['0', 'X', '0', 'Y', '0', 'X', '0', '0'],
        ['0', '0', '0', 'X', '0', '0', '0', '0'],
        ['X', '0', 'Y', '0', '0', '0', 'X', '0'],
        ['X', '0', 'B', 'B', 'B', 'B', 'X', '0'],
        ['X', '0', 'B', 'B', 'B', 'B', 'Y', '0'],
        ['0', '0', '0', 'Y', '0', '0', '0', '0']
    ]
}

class Config:
    # ========== CONFIGURAÇÕES OTIMIZADAS ==========

    # MAPPO específico
    ACTOR_LR = 1e-4          # Reduzido para mais estabilidade (antes 3e-4)
    CRITIC_LR = 1e-4         # Reduzido para mais estabilidade
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENTROPY_COEF = 0.05      # Aumentado para mais exploração (antes 0.01)
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 20          # Mais épocas para melhor convergência (antes 10)
    BATCH_SIZE = 32          # Batch menor para mais atualizações (antes 64)
    MINI_BATCH_SIZE = 16

    # Exploração - AUMENTADA
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 100000  # Aumentado (antes 50000)

    # Treinamento
    MAX_STEPS = 500
    EPISODES_PER_SESSION = 3000   # 3000 episódios
    LEARNING_STARTS = 1000

    # Sistema
    SAVE_INTERVAL = 100
    CLEAN_MEMORY_EVERY = 500

    # Barreiras Y (probabilidade de serem removidas)
    Y_BARRIER_REMOVAL_PROB = 0.5

    # Falhas dos robôs
    FAILURE_PROBABILITY = 0.2
    FAILURE_PENALTY = -0.15


# ==================== REDE NEURAL PARA MAPPO ====================
class ActorNetwork(nn.Module):
    """Rede de política (ator) para cada agente"""
    def __init__(self, input_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x):
        logits = self.network(x)
        return F.softmax(logits, dim=-1), logits


class CriticNetwork(nn.Module):
    """Rede de valor (crítico) centralizado"""
    def __init__(self, global_state_dim, hidden_dim=256):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


# ==================== AGENTE MAPPO ====================
class MAPPOAgent:
    def __init__(self, agent_id, state_dim, action_dim, config, global_state_dim):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.global_state_dim = global_state_dim
        self.device = _pick_device()

        # Redes do agente
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(global_state_dim).to(self.device)

        # Otimizadores
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # Memória para trajetórias
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.global_states = []

        self.steps_done = 0
        self.epsilon = config.EPSILON_START

    def get_epsilon(self):
        # Decaimento linear do epsilon
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END

        epsilon = self.config.EPSILON_START - (self.config.EPSILON_START - self.config.EPSILON_END) * \
                  self.steps_done / self.config.EPSILON_DECAY_STEPS
        return max(self.config.EPSILON_END, epsilon)

    def decay_epsilon(self):
        # O epsilon já é calculado com base nos steps_done
        pass

    def select_action(self, state, global_state, training=True):
        self.steps_done += 1

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs, _ = self.actor(state_tensor)
            # CORRECAO DO BUG: PPO precisa AMOSTRAR da distribuicao da politica e
            # registrar o log_prob DESSA amostragem (nao usar epsilon-greedy + argmax,
            # que invalida a razao de importancia do PPO). A exploracao vem da
            # politica estocastica + bonus de entropia.
            dist = torch.distributions.Categorical(probs)
            if training:
                action = dist.sample()
            else:
                action = probs.argmax(dim=-1)
            log_prob = dist.log_prob(action)
            return action.item(), log_prob.item()

    def store_transition(self, state, action, log_prob, reward, done, global_state):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.global_states.append(global_state)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.global_states = []

    def update(self):
        """Atualiza as redes usando PPO"""
        if len(self.states) == 0:
            return 0, 0

        # Converter para tensores
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        global_states = torch.FloatTensor(np.array(self.global_states)).to(self.device)

        # Calcular valores e vantagens
        with torch.no_grad():
            values = self.critic(global_states).squeeze()
            advantages = self.compute_gae(rewards, values, dones)
            returns = advantages + values

        # Normalizar vantagens
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO updates
        total_actor_loss = 0
        total_critic_loss = 0
        n_updates = 0

        for _ in range(self.config.PPO_EPOCHS):
            # Amostrar mini-batches
            indices = np.random.permutation(len(self.states))

            for start in range(0, len(indices), self.config.MINI_BATCH_SIZE):
                end = start + self.config.MINI_BATCH_SIZE
                batch_indices = indices[start:end]

                # Batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_global_states = global_states[batch_indices]

                # Actor loss
                probs, _ = self.actor(batch_states)
                new_log_probs = torch.log(probs.gather(1, batch_actions.unsqueeze(1)).squeeze() + 1e-10)

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                actor_loss = actor_loss - self.config.ENTROPY_COEF * entropy

                # Critic loss
                values = self.critic(batch_global_states).squeeze()
                critic_loss = F.mse_loss(values, batch_returns)

                # Backpropagation
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.MAX_GRAD_NORM)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.MAX_GRAD_NORM)
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                n_updates += 1

        # Limpar memória
        self.clear_memory()

        return total_actor_loss / max(1, n_updates), total_critic_loss / max(1, n_updates)

    def compute_gae(self, rewards, values, dones):
        """Computa Generalized Advantage Estimation"""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.config.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.GAMMA * self.config.LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return torch.FloatTensor(advantages).to(self.device)


# ==================== AMBIENTE WAREHOUSE ====================
class WarehouseEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, config=None):
        super().__init__()

        self.config = config or Config()
        self.height = MAP_CONFIG['height']
        self.width = MAP_CONFIG['width']
        self.base_grid = [row[:] for row in MAP_CONFIG['grid']]
        self.grid = [row[:] for row in self.base_grid]

        # Encontrar posições de Y
        self.y_positions = self._find_positions('Y')

        self.robot_positions = None
        self.box_positions = None
        self.targets = self._find_positions('B')

        self.num_robots = 2
        self.num_boxes = len(self._find_positions('A'))
        self.num_targets = len(self.targets)

        self.delivered_boxes = None
        self.steps = 0
        self.max_steps = self.config.MAX_STEPS
        self.total_deliveries = 0
        self.collisions = 0
        self.failures = [0, 0]
        self.distance_traveled = [0, 0]

        self.action_space = spaces.Tuple([spaces.Discrete(6) for _ in range(self.num_robots)])

        obs_dim = (self.num_robots * 2) + (self.num_boxes * 2) + (self.num_targets * 2) + 8
        self.observation_space = spaces.Box(
            low=-1, high=self.height + self.width,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Global state dimension para o crítico centralizado
        self.global_state_dim = (self.num_robots * 2) + (self.num_boxes * 2) + self.num_boxes + 4

    def _find_positions(self, symbols):
        if isinstance(symbols, str):
            symbols = [symbols]
        positions = []
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid[i][j]
                if any(cell.startswith(sym) for sym in symbols):
                    positions.append((i, j))
        return positions

    def _remove_random_y_barriers(self):
        """Remove aleatoriamente as barreiras Y com probabilidade configurada"""
        self.grid = [row[:] for row in self.base_grid]

        for y_pos in self.y_positions:
            if random.random() < self.config.Y_BARRIER_REMOVAL_PROB:
                i, j = y_pos
                self.grid[i][j] = '0'

    def reset(self):
        self._remove_random_y_barriers()

        self.robot_positions = self._find_positions('R')
        self.box_positions = self._find_positions('A')
        self.delivered_boxes = [False] * self.num_boxes
        self.targets = self._find_positions('B')

        self.steps = 0
        self.total_deliveries = 0
        self.collisions = 0
        self.failures = [0, 0]
        self.distance_traveled = [0, 0]

        self.previous_distances = [self._min_distance_to_boxes(r) for r in range(self.num_robots)]

        return self._get_observation(), self._get_info()

    def _min_distance_to_boxes(self, robot_id):
        robot_pos = self.robot_positions[robot_id]
        remaining_boxes = [box_pos for box_pos in self.box_positions
                          if box_pos is not None and
                          not self.delivered_boxes[self.box_positions.index(box_pos)]]

        if not remaining_boxes:
            return 0
        return min([abs(robot_pos[0] - box_pos[0]) + abs(robot_pos[1] - box_pos[1])
                   for box_pos in remaining_boxes])

    def _is_valid_position(self, pos, robot_id=None):
        i, j = pos
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return False

        cell = self.grid[i][j]
        if cell == 'X':
            return False

        if robot_id is not None:
            for rid, rpos in enumerate(self.robot_positions):
                if rid != robot_id and rpos == (i, j):
                    return False
        return True

    def _get_alternative_direction(self, original_action, robot_id):
        i, j = self.robot_positions[robot_id]
        alternative_actions = [a for a in range(4) if a != original_action]
        random.shuffle(alternative_actions)

        for alt_action in alternative_actions:
            alt_i, alt_j = i, j
            if alt_action == 0: alt_i -= 1
            elif alt_action == 1: alt_i += 1
            elif alt_action == 2: alt_j -= 1
            elif alt_action == 3: alt_j += 1

            if self._is_valid_position((alt_i, alt_j), robot_id):
                return alt_action, (alt_i, alt_j)

        return None, None

    def _move_robot_with_failure(self, robot_id, action):
        i, j = self.robot_positions[robot_id]
        new_i, new_j = i, j

        if action == 0: new_i -= 1
        elif action == 1: new_i += 1
        elif action == 2: new_j -= 1
        elif action == 3: new_j += 1

        desired_valid = self._is_valid_position((new_i, new_j), robot_id)

        if random.random() < self.config.FAILURE_PROBABILITY:
            self.failures[robot_id] += 1

            alt_action, alt_pos = self._get_alternative_direction(action, robot_id)
            if alt_action is not None:
                old_pos = self.robot_positions[robot_id]
                self.grid[old_pos[0]][old_pos[1]] = '0'
                self.grid[alt_pos[0]][alt_pos[1]] = f'R{robot_id + 1}'
                self.robot_positions[robot_id] = alt_pos
                self.distance_traveled[robot_id] += 1
                return self.config.FAILURE_PENALTY - 0.01

            return self.config.FAILURE_PENALTY

        if desired_valid:
            self.distance_traveled[robot_id] += 1
            old_pos = self.robot_positions[robot_id]
            self.grid[old_pos[0]][old_pos[1]] = '0'
            self.grid[new_i][new_j] = f'R{robot_id + 1}'
            self.robot_positions[robot_id] = (new_i, new_j)
            return -0.005
        else:
            self.collisions += 1
            return -0.02

    def _pickup_box(self, robot_id):
        robot_pos = self.robot_positions[robot_id]
        for box_id, box_pos in enumerate(self.box_positions):
            if not self.delivered_boxes[box_id] and box_pos == robot_pos:
                self.box_positions[box_id] = None
                self.grid[robot_pos[0]][robot_pos[1]] = f'R{robot_id + 1}'
                return 2.0
        return -0.02

    def _drop_box(self, robot_id):
        robot_pos = self.robot_positions[robot_id]

        box_with_robot = None
        for box_id, box_pos in enumerate(self.box_positions):
            if box_pos is None and not self.delivered_boxes[box_id]:
                box_with_robot = box_id
                break

        if box_with_robot is None:
            return -0.02

        for target_pos in self.targets:
            if robot_pos == target_pos:
                self.delivered_boxes[box_with_robot] = True
                self.total_deliveries += 1
                self.grid[robot_pos[0]][robot_pos[1]] = f'R{robot_id + 1}'
                return 25.0

        return -0.05

    def _calculate_shaped_reward(self, robot_id, base_reward):
        reward = base_reward

        current_distance = self._min_distance_to_boxes(robot_id)
        previous_distance = self.previous_distances[robot_id]

        if current_distance < previous_distance:
            reward += 0.1 * (previous_distance - current_distance)
        elif current_distance > previous_distance:
            reward -= 0.02 * (current_distance - previous_distance)

        self.previous_distances[robot_id] = current_distance

        if all(self.delivered_boxes):
            reward += 50.0

        return reward

    def _get_observation(self):
        obs = []

        for robot_pos in self.robot_positions:
            obs.append(robot_pos[0] / self.height)
            obs.append(robot_pos[1] / self.width)

        for box_id, box_pos in enumerate(self.box_positions):
            if box_pos is None or self.delivered_boxes[box_id]:
                obs.append(-1)
                obs.append(-1)
            else:
                obs.append(box_pos[0] / self.height)
                obs.append(box_pos[1] / self.width)

        for target_pos in self.targets:
            obs.append(target_pos[0] / self.height)
            obs.append(target_pos[1] / self.width)

        for robot_pos in self.robot_positions:
            min_box_dist = min([abs(robot_pos[0] - box_pos[0]) + abs(robot_pos[1] - box_pos[1])
                               for box_pos in self.box_positions
                               if box_pos is not None and
                               not self.delivered_boxes[self.box_positions.index(box_pos)]],
                              default=100)
            obs.append(min_box_dist / (self.height + self.width))

            min_target_dist = min([abs(robot_pos[0] - target_pos[0]) + abs(robot_pos[1] - target_pos[1])
                                  for target_pos in self.targets],
                                 default=100)
            obs.append(min_target_dist / (self.height + self.width))

        return np.array(obs, dtype=np.float32)

    def get_global_state(self):
        global_state = []

        for robot_pos in self.robot_positions:
            global_state.append(robot_pos[0] / self.height)
            global_state.append(robot_pos[1] / self.width)

        for box_pos in self.box_positions:
            if box_pos is not None:
                global_state.append(box_pos[0] / self.height)
                global_state.append(box_pos[1] / self.width)
            else:
                global_state.append(-1)
                global_state.append(-1)

        for delivered in self.delivered_boxes:
            global_state.append(1.0 if delivered else 0.0)

        global_state.append(self.steps / self.max_steps)
        global_state.append(self.total_deliveries / self.num_boxes)

        return np.array(global_state, dtype=np.float32)

    def step(self, actions):
        self.steps += 1

        if len(actions) != self.num_robots:
            actions = [actions] * self.num_robots

        total_reward = 0
        rewards = [0, 0]

        movement_rewards = []
        for robot_id, action in enumerate(actions):
            if action < 4:
                reward = self._move_robot_with_failure(robot_id, action)
                movement_rewards.append(reward)
            else:
                movement_rewards.append(0)

        interaction_rewards = []
        for robot_id, action in enumerate(actions):
            if action == 4:
                reward = self._pickup_box(robot_id)
                interaction_rewards.append(reward)
            elif action == 5:
                reward = self._drop_box(robot_id)
                interaction_rewards.append(reward)
            else:
                interaction_rewards.append(0)

        for robot_id in range(self.num_robots):
            base_reward = movement_rewards[robot_id] + interaction_rewards[robot_id]
            shaped_reward = self._calculate_shaped_reward(robot_id, base_reward)
            rewards[robot_id] = shaped_reward
            total_reward += shaped_reward

        terminated = all(self.delivered_boxes)
        truncated = self.steps >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, rewards, terminated, truncated, info

    def _get_info(self):
        return {
            'steps': self.steps,
            'total_deliveries': self.total_deliveries,
            'collisions': self.collisions,
            'failures_r1': self.failures[0],
            'failures_r2': self.failures[1],
            'distance_traveled': self.distance_traveled.copy(),
            'remaining_boxes': sum(1 for d in self.delivered_boxes if not d),
            'success_rate': self.total_deliveries / self.num_boxes if self.steps > 0 else 0,
        }

    def close(self):
        pass


# ==================== FUNÇÕES DE PLOTAGEM (SEM MÉDIA MÓVEL) ====================
def plot_results(metrics, save_dir, window=100):
    """Plota e salva todos os gráficos COM média móvel de `window` episódios."""

    print(f"\n📊 Gerando gráficos (com média móvel de {window}) em: {save_dir}")

    episodes = range(1, len(metrics['episode_rewards']) + 1)

    def _ma(y):
        """Retorna (x, média_móvel) para a janela; vazio se dados insuficientes."""
        y = np.asarray(y, dtype=float)
        if len(y) < window:
            return [], []
        ma = np.convolve(y, np.ones(window) / window, mode='valid')
        x = range(window, len(y) + 1)
        return x, ma

    def _grafico(y, cor, ylabel, titulo, filename, *, ylim=None, hlines=None):
        plt.figure(figsize=(14, 7))
        # série bruta (fraca) + média móvel (forte)
        plt.plot(episodes, y, color=cor, alpha=0.25, linewidth=0.8, label='Por episódio')
        x_ma, ma = _ma(y)
        if len(ma):
            plt.plot(x_ma, ma, 'r-', linewidth=2.2, label=f'Média móvel ({window})')
        for hy, hcolor, hlabel in (hlines or []):
            plt.axhline(y=hy, color=hcolor, linestyle='--', linewidth=2, label=hlabel)
        plt.xlabel('Episódio', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(titulo, fontsize=14, fontweight='bold')
        if ylim:
            plt.ylim(ylim)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {filename} salvo")

    _grafico(metrics['episode_rewards'], 'b', 'Recompensa',
             'Recompensa por Episódio (3000 episódios) - MAPPO',
             'grafico_recompensa.png')

    _grafico(metrics['episode_deliveries'], 'g', 'Entregas',
             'Entregas por Episódio (3000 episódios) - MAPPO',
             'grafico_entregas.png', ylim=[0, 9],
             hlines=[(8, 'orange', 'Meta (8 caixas)')])

    _grafico(metrics['episode_steps'], 'orange', 'Steps',
             'Steps por Episódio (3000 episódios) - MAPPO',
             'grafico_steps.png')

    _grafico(metrics['success_rates'], 'purple', 'Taxa de Sucesso',
             'Taxa de Sucesso por Episódio (3000 episódios) - MAPPO',
             'grafico_taxa_sucesso.png', ylim=[0, 1.05],
             hlines=[(0.95, 'green', 'Meta 95%')])

    _grafico(metrics['collisions'], 'red', 'Colisões',
             'Colisões por Episódio (3000 episódios) - MAPPO',
             'grafico_colisoes.png')

    # Falhas: total + R1 + R2, com média móvel do total
    plt.figure(figsize=(14, 7))
    failures_total = np.array(metrics['failures_r1']) + np.array(metrics['failures_r2'])
    plt.plot(episodes, failures_total, 'brown', alpha=0.25, linewidth=0.8, label='Total')
    plt.plot(episodes, metrics['failures_r1'], 'red', alpha=0.2, linewidth=0.6, label='R1')
    plt.plot(episodes, metrics['failures_r2'], 'blue', alpha=0.2, linewidth=0.6, label='R2')
    x_ma, ma = _ma(failures_total)
    if len(ma):
        plt.plot(x_ma, ma, 'k-', linewidth=2.2, label=f'Média móvel total ({window})')
    plt.xlabel('Episódio', fontsize=12)
    plt.ylabel('Falhas', fontsize=12)
    plt.title('Falhas por Episódio (3000 episódios) - MAPPO', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'grafico_falhas.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ grafico_falhas.png salvo")

    print(f"📊 Todos os 6 gráficos foram salvos em: {save_dir}")


def save_metrics_csv(metrics, save_dir):
    """Salva as métricas em arquivo CSV"""
    df = pd.DataFrame({
        'episodio': range(1, len(metrics['episode_rewards']) + 1),
        'recompensa': metrics['episode_rewards'],
        'entregas': metrics['episode_deliveries'],
        'steps': metrics['episode_steps'],
        'taxa_sucesso': metrics['success_rates'],
        'colisoes': metrics['collisions'],
        'falha_r1': metrics['failures_r1'],
        'falha_r2': metrics['failures_r2']
    })
    df.to_csv(save_dir / 'metricas.csv', index=False)
    print(f"📊 Métricas salvas em: {save_dir / 'metricas.csv'}")


# ==================== FUNÇÃO DE TREINAMENTO MAPPO ====================
def train_mappo(num_episodes=3000):
    """Treina os agentes usando MAPPO por 3000 episódios"""

    config = Config()
    env = WarehouseEnv(config=config)

    # Dimensões
    sample_obs, _ = env.reset()
    state_dim = len(sample_obs)
    action_dim = 6
    global_state_dim = len(env.get_global_state())
    env.close()

    print("=" * 80)
    print("🏭 TREINAMENTO MAPPO - WAREHOUSE COM BARREIRAS Y REMOVÍVEIS")
    print("=" * 80)
    print(f"\n📋 CONFIGURAÇÃO OTIMIZADA:")
    print(f"   • Algoritmo: MAPPO (Multi-Agent PPO)")
    print(f"   • Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   • Total de episódios: {num_episodes}")
    print(f"   • EPSILON_DECAY_STEPS: {config.EPSILON_DECAY_STEPS}")
    print(f"   • ACTOR_LR: {config.ACTOR_LR}")
    print(f"   • ENTROPY_COEF: {config.ENTROPY_COEF}")
    print(f"   • PPO_EPOCHS: {config.PPO_EPOCHS}")
    print(f"   • BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   • Barreiras Y removidas: {config.Y_BARRIER_REMOVAL_PROB*100:.0f}% chance")
    print(f"   • Falhas nos robôs: {config.FAILURE_PROBABILITY*100:.0f}% chance")
    print("=" * 80)

    # Criar agentes
    agents = [MAPPOAgent(i, state_dim, action_dim, config, global_state_dim) for i in range(2)]

    metrics = {
        'episode_rewards': [],
        'episode_deliveries': [],
        'episode_steps': [],
        'success_rates': [],
        'collisions': [],
        'failures_r1': [],
        'failures_r2': []
    }

    # Criar diretório de resultados em LocalMultiAgente/resultados/mappo_corrigido
    results_dir = Path(__file__).resolve().parent.parent / "resultados" / "mappo_corrigido"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Diretório de resultados: {results_dir.absolute()}")

    total_start_time = time.time()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_collisions = 0

        # Limpar memórias dos agentes no início do episódio
        for agent in agents:
            agent.clear_memory()

        global_state = env.get_global_state()

        for step in range(config.MAX_STEPS):
            # Selecionar ações
            actions = []
            log_probs = []

            for agent in agents:
                action, log_prob = agent.select_action(obs, global_state, training=True)
                actions.append(action)
                log_probs.append(log_prob)

            # Executar ações
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_global_state = env.get_global_state()

            # Armazenar transições
            for i, agent in enumerate(agents):
                agent.store_transition(obs, actions[i], log_probs[i], rewards[i],
                                      terminated or truncated, global_state)

            episode_reward += sum(rewards)
            episode_collisions = info['collisions']

            obs = next_obs
            global_state = next_global_state

            if terminated or truncated:
                break

        # Atualizar agentes
        for agent in agents:
            agent.update()

        # Registrar métricas
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_deliveries'].append(info['total_deliveries'])
        metrics['episode_steps'].append(step + 1)
        metrics['success_rates'].append(info['success_rate'])
        metrics['collisions'].append(episode_collisions)
        metrics['failures_r1'].append(info['failures_r1'])
        metrics['failures_r2'].append(info['failures_r2'])

        # Logging a cada 100 episódios
        if (episode + 1) % 100 == 0:
            # Calcular estatísticas dos últimos 100 episódios
            recent_rewards = metrics['episode_rewards'][-100:]
            recent_deliveries = metrics['episode_deliveries'][-100:]
            epsilon = agents[0].get_epsilon()
            elapsed = time.time() - total_start_time

            print(f"Ep {episode+1:4d}/{num_episodes} | "
                  f"Reward: {np.mean(recent_rewards):7.2f} | "
                  f"Entregas: {np.mean(recent_deliveries):.2f}/8 | "
                  f"ε: {epsilon:.3f} | "
                  f"Tempo: {elapsed/60:.1f}min")

        # Limpar memória periodicamente
        if (episode + 1) % config.CLEAN_MEMORY_EVERY == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = (time.time() - total_start_time) / 60
    env.close()

    print(f"\n💾 SALVANDO RESULTADOS...")

    # Salvar métricas
    save_metrics_csv(metrics, results_dir)

    # Plotar gráficos (sem média móvel)
    plot_results(metrics, results_dir)

    # Salvar modelos
    models_dir = results_dir / "models"
    models_dir.mkdir(exist_ok=True)
    for i, agent in enumerate(agents):
        torch.save(agent.actor.state_dict(), models_dir / f"mappo_actor_{i}_3000ep.pth")
        torch.save(agent.critic.state_dict(), models_dir / f"mappo_critic_{i}_3000ep.pth")

    # Calcular estatísticas finais
    final_deliveries = metrics['episode_deliveries'][-100:]
    final_rewards = metrics['episode_rewards'][-100:]
    final_collisions = metrics['collisions'][-100:]

    # Relatório final
    report = f"""
    ========================================
    RELATÓRIO FINAL - MAPPO (3000 EPISÓDIOS)
    ========================================

    DATA: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    DIRETÓRIO: {results_dir.absolute()}

    CONFIGURAÇÃO OTIMIZADA:
    - Total de episódios: {num_episodes}
    - Tempo total: {total_time:.1f} minutos
    - EPSILON_DECAY_STEPS: {config.EPSILON_DECAY_STEPS}
    - ACTOR_LR: {config.ACTOR_LR}
    - ENTROPY_COEF: {config.ENTROPY_COEF}
    - PPO_EPOCHS: {config.PPO_EPOCHS}
    - Barreiras Y removidas: {config.Y_BARRIER_REMOVAL_PROB*100:.0f}% chance
    - Falhas nos robôs: {config.FAILURE_PROBABILITY*100:.0f}% chance

    MÉTRICAS FINAIS (últimos 100 episódios):
    - Recompensa média: {np.mean(final_rewards):.2f}
    - Entregas médias: {np.mean(final_deliveries):.2f}/8
    - Taxa de sucesso final: {metrics['success_rates'][-1]:.1%}
    - Colisões médias: {np.mean(final_collisions):.1f}
    - Total falhas R1: {sum(metrics['failures_r1'])}
    - Total falhas R2: {sum(metrics['failures_r2'])}

    MELHORES RESULTADOS:
    - Melhor recompensa: {max(metrics['episode_rewards']):.2f}
    - Melhor entrega: {max(metrics['episode_deliveries'])}/8
    - Menor número de steps: {min(metrics['episode_steps'])}

    GRÁFICOS GERADOS (SEM MÉDIA MÓVEL):
    - grafico_recompensa.png (3000 pontos)
    - grafico_entregas.png (3000 pontos)
    - grafico_steps.png (3000 pontos)
    - grafico_taxa_sucesso.png (3000 pontos)
    - grafico_colisoes.png (3000 pontos)
    - grafico_falhas.png (3000 pontos)

    ========================================
    """

    with open(results_dir / "relatorio.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n✅ TREINAMENTO MAPPO CONCLUÍDO!")
    print(f"📁 Resultados salvos em: {results_dir.absolute()}")
    print(f"   - 6 gráficos (sem média móvel)")
    print(f"   - CSV com todos os 3000 episódios")
    print(f"   - Modelos treinados")

    return agents, metrics, results_dir



# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MAPPO corrigido (standalone)")
    parser.add_argument("--episodes", type=int, default=1500,
                        help="numero de episodios de treino (default 1500)")
    parser.add_argument("--device", type=str, default=None,
                        help="forca dispositivo: cpu | mps | cuda")
    parser.add_argument("--ppo-epochs", type=int, default=None,
                        help="override Config.PPO_EPOCHS (default do notebook = 20)")
    parser.add_argument("--minibatch", type=int, default=None,
                        help="override Config.MINI_BATCH_SIZE (default do notebook = 16)")
    args = parser.parse_args()

    if args.device:
        _os.environ["MAPPO_DEVICE"] = args.device
    if args.ppo_epochs is not None:
        Config.PPO_EPOCHS = args.ppo_epochs
    if args.minibatch is not None:
        Config.MINI_BATCH_SIZE = args.minibatch

    print("Dispositivo:", _pick_device(),
          "| PPO_EPOCHS:", Config.PPO_EPOCHS,
          "| MINI_BATCH:", Config.MINI_BATCH_SIZE)
    t0 = time.time()
    agents, metrics, results_dir = train_mappo(num_episodes=args.episodes)
    dt = time.time() - t0
    n = len(metrics["episode_rewards"])
    print(f"\n[TIMING] {n} episodios em {dt/60:.2f} min "
          f"({dt/max(1,n):.3f} s/episodio). "
          f"Estimativa p/ 3000 ep: {dt/max(1,n)*3000/60:.1f} min.")
