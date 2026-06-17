#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VDN AJUSTADO (standalone, roda local) — INF2072 / PUC-DI.

Copia fiel do notebook `Executa Experimento VDN.ipynb` com DOIS ajustes:
  1. MAX_STEPS = 500 (alinhado com IDQN/MAPPO; o original usava 1000).
  2. ID DO AGENTE na observacao: cada rede recebe a obs global + um one-hot do
     seu indice. Isso quebra a simetria (as duas redes recebiam input identico),
     o principal suspeito de fazer os dois robos disputarem a mesma caixa.

Objetivo: ver se o VDN melhora com esses ajustes. O loss VDN em si ja estava
correto (Q_total = Q1+Q2, Double-DQN por agente, PER) — nao mexemos nele.

Uso:
    python vdn_ajustado.py --episodes 30                 # probe de tempo
    python vdn_ajustado.py --episodes 3000 --device cpu  # treino completo
"""
import argparse
import os as _os


def _pick_device():
    import torch
    forced = _os.environ.get("VDN_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


"""
IMPLEMENTAÇÃO VDN - VERSÃO 1.0.0
- Value Decomposition Networks para 2 robôs
- 3000 episódios de treinamento
- Gráficos individuais separados
- Aprendizado centralizado com execução descentralizada
- Q_total = Q_1 + Q_2 (decomposição aditiva)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import warnings
import time
import gc

warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO ====================
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


class VDNConfig:
    # ── Exploração ──────────────────────────────────────────────
    EPSILON_START        = 1.0
    EPSILON_END          = 0.02          # mais baixo → mais exploração determinística no final
    EPSILON_DECAY_STEPS  = 120000        # decai ao longo de ~2/3 dos steps totais

    # ── Aprendizado ─────────────────────────────────────────────
    LEARNING_RATE        = 0.0003        # Adam com lr levemente maior beneficia VDN
    BATCH_SIZE           = 256
    GAMMA                = 0.97          # horizonte mais longo melhora planejamento
    TAU                  = 0.005         # soft-update suave
    WEIGHT_DECAY         = 1e-5

    # ── Rede Neural ─────────────────────────────────────────────
    HIDDEN_DIM           = 256           # suficiente; evita overfitting em mapas pequenos
    DROPOUT_RATE         = 0.1

    # ── Memória ─────────────────────────────────────────────────
    BUFFER_SIZE          = 200000        # buffer maior para cobrir 3000 episódios
    ALPHA                = 0.6           # priorização
    BETA_START           = 0.4
    BETA_END             = 1.0           # annealing completo até o final

    # ── Treinamento ─────────────────────────────────────────────
    MAX_GRAD_NORM        = 10.0
    LEARNING_STARTS      = 2000          # aguarda buffer razoável antes de treinar
    TRAIN_FREQ           = 2             # treina a cada 2 steps → mais atualizações
    TARGET_UPDATE_FREQ   = 200           # hard-update da target a cada 200 steps de aprendizado
    USE_SOFT_UPDATE      = True
    MAX_STEPS            = 500           # reduz steps máx → episódios mais rápidos
    EPISODES_TOTAL       = 3000

    # ── Sistema ─────────────────────────────────────────────────
    SAVE_INTERVAL        = 500
    CLEAN_MEMORY_EVERY   = 500

    # ── Falhas dos robôs ────────────────────────────────────────
    FAILURE_PROBABILITY      = 0.15      # ligeiramente menor → ambiente menos aleatório
    FAILURE_PENALTY          = -0.10
    ALTERNATIVE_DIRECTIONS   = True


# ==================== PRIORITIZED REPLAY BUFFER ====================
class PrioritizedReplayBuffer:
    """Buffer de replay com priorização por TD-error."""

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity  = capacity
        self.alpha     = alpha
        self.buffer    = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position  = 0
        self._max_priority = 1.0

    def push(self, state, actions, rewards, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, actions, rewards, next_state, done)
        self.priorities[self.position] = self._max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        size = len(self.buffer)
        probs = self.priorities[:size]
        probs = probs / probs.sum()

        indices = np.random.choice(size, batch_size, replace=False, p=probs)
        weights = (size * probs[indices]) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                indices,
                weights.astype(np.float32))

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = (abs(err) + 1e-6) ** self.alpha
            self.priorities[idx] = p
            if p > self._max_priority:
                self._max_priority = p

    def __len__(self):
        return len(self.buffer)


# ==================== REDE NEURAL INDIVIDUAL (Q_i) ====================
class AgentNet(nn.Module):
    """Rede Q individual para cada agente (mesma arquitetura, pesos separados)."""

    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 256, dropout: float = 0.1):
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
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # última camada com gain menor → outputs inicialmente próximos de 0
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ==================== VDN CONTROLLER ====================
class VDNController:
    """
    Controlador VDN centralizado.
    Q_total = Q_1(o_1, a_1) + Q_2(o_2, a_2)
    Treina ambas as redes com um único otimizador e loss compartilhado.
    """

    def __init__(self, n_agents: int, state_dim: int, action_dim: int,
                 config: VDNConfig):
        self.n_agents   = n_agents
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.config     = config
        self.device     = _pick_device()

        print(f"  VDN Controller usando dispositivo: {self.device}")

        # Redes policy e target para cada agente
        self.policy_nets = nn.ModuleList([
            AgentNet(state_dim + n_agents, action_dim, config.HIDDEN_DIM, config.DROPOUT_RATE)
            for _ in range(n_agents)
        ]).to(self.device)

        self.target_nets = nn.ModuleList([
            AgentNet(state_dim + n_agents, action_dim, config.HIDDEN_DIM, config.DROPOUT_RATE)
            for _ in range(n_agents)
        ]).to(self.device)

        for i in range(n_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())
            self.target_nets[i].eval()

        # Otimizador único para todos os parâmetros
        self.optimizer = optim.Adam(
            self.policy_nets.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPISODES_TOTAL, eta_min=1e-5
        )

        self.memory         = PrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.ALPHA)
        self.steps_done     = 0
        self.learning_steps = 0
        self.losses         = []

    # ── Beta annealing ────────────────────────────────────────────────────────
    def _get_beta(self) -> float:
        frac = min(1.0, self.steps_done / (self.config.EPISODES_TOTAL * self.config.MAX_STEPS))
        return self.config.BETA_START + frac * (self.config.BETA_END - self.config.BETA_START)

    # ── Epsilon ────────────────────────────────────────────────────────────────
    def get_epsilon(self) -> float:
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END
        t = self.steps_done / self.config.EPSILON_DECAY_STEPS
        return self.config.EPSILON_START + t * (self.config.EPSILON_END - self.config.EPSILON_START)

    # ── Seleção de ação ────────────────────────────────────────────────────────
    def _aug(self, states, agent_idx):
        """Concatena one-hot do ID do agente a cada linha (quebra a simetria)."""
        n = states.shape[0]
        onehot = torch.zeros(n, self.n_agents, device=states.device)
        onehot[:, agent_idx] = 1.0
        return torch.cat([states, onehot], dim=1)

    def select_actions(self, obs: np.ndarray, training: bool = True) -> List[int]:
        """obs: observação global compartilhada (shape: [obs_dim])"""
        self.steps_done += 1
        eps = self.get_epsilon() if training else 0.0
        actions = []
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            for i, net in enumerate(self.policy_nets):
                if training and random.random() < eps:
                    actions.append(random.randrange(self.action_dim))
                else:
                    q = net(self._aug(obs_t, i))
                    actions.append(q.argmax(dim=1).item())
        return actions

    # ── Armazenar transição ───────────────────────────────────────────────────
    def remember(self, state, actions, rewards, next_state, done):
        self.memory.push(state, actions, rewards, next_state, done)

    # ── Otimização VDN ────────────────────────────────────────────────────────
    def optimize(self) -> float:
        if (len(self.memory) < self.config.BATCH_SIZE or
                self.steps_done < self.config.LEARNING_STARTS):
            return 0.0

        self.learning_steps += 1
        if self.learning_steps % self.config.TRAIN_FREQ != 0:
            return 0.0

        beta = self._get_beta()
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.config.BATCH_SIZE, beta=beta)

        S  = torch.FloatTensor(states).to(self.device)       # [B, obs]
        A  = torch.LongTensor(actions).to(self.device)       # [B, n_agents]
        R  = torch.FloatTensor(rewards).to(self.device)      # [B, n_agents]
        S_ = torch.FloatTensor(next_states).to(self.device)  # [B, obs]
        D  = torch.FloatTensor(dones).to(self.device)        # [B]
        W  = torch.FloatTensor(weights).to(self.device)      # [B]

        # ── Q_total atual ────────────────────────────────────────
        q_total = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        for i, net in enumerate(self.policy_nets):
            q_vals = net(self._aug(S, i))                     # [B, n_actions]
            q_a    = q_vals.gather(1, A[:, i].unsqueeze(1)).squeeze(1)
            q_total = q_total + q_a

        # ── Q_total target (Double DQN per agent) ────────────────
        with torch.no_grad():
            q_total_next = torch.zeros(self.config.BATCH_SIZE, device=self.device)
            for i, (pnet, tnet) in enumerate(zip(self.policy_nets, self.target_nets)):
                next_a  = pnet(self._aug(S_, i)).argmax(dim=1, keepdim=True)  # greedy policy
                next_q  = tnet(self._aug(S_, i)).gather(1, next_a).squeeze(1)  # valor target
                q_total_next = q_total_next + next_q

            r_total = R.sum(dim=1)
            y = r_total + self.config.GAMMA * q_total_next * (1 - D)

        # ── Loss ponderada ───────────────────────────────────────
        td_errors = (y - q_total).detach().cpu().numpy()
        loss = (W * (y - q_total).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_nets.parameters(), self.config.MAX_GRAD_NORM)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        # ── Atualização da target ─────────────────────────────────
        if self.config.USE_SOFT_UPDATE:
            self._soft_update()
        elif self.learning_steps % self.config.TARGET_UPDATE_FREQ == 0:
            self._hard_update()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def _soft_update(self):
        tau = self.config.TAU
        for pnet, tnet in zip(self.policy_nets, self.target_nets):
            for tp, pp in zip(tnet.parameters(), pnet.parameters()):
                tp.data.copy_(tau * pp.data + (1 - tau) * tp.data)

    def _hard_update(self):
        for pnet, tnet in zip(self.policy_nets, self.target_nets):
            tnet.load_state_dict(pnet.state_dict())

    def save(self, save_dir: Path, episode: int):
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(self.policy_nets):
            torch.save(net.state_dict(), save_dir / f"vdn_agent{i}_ep{episode}.pth")

    def save_final(self, save_dir: Path):
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, net in enumerate(self.policy_nets):
            torch.save(net.state_dict(), save_dir / f"vdn_agent{i}_final.pth")


# ==================== AMBIENTE WAREHOUSE ====================
class WarehouseEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, config: VDNConfig = None):
        super().__init__()
        self.config    = config or VDNConfig()
        self.height    = MAP_CONFIG['height']
        self.width     = MAP_CONFIG['width']
        self.base_grid = [row[:] for row in MAP_CONFIG['grid']]
        self.grid      = [row[:] for row in self.base_grid]

        self.y_positions  = self._find_positions('Y')
        self.num_robots   = 2
        self.num_boxes    = len(self._find_positions('A'))
        self.num_targets  = len(self._find_positions('B'))

        self.robot_positions  = None
        self.box_positions    = None
        self.delivered_boxes  = None
        self.targets          = None
        self.steps            = 0
        self.max_steps        = self.config.MAX_STEPS
        self.total_deliveries = 0
        self.collisions       = 0
        self.failures         = [0, 0]
        self.distance_traveled = [0, 0]
        self.previous_distances = None

        self.action_space = spaces.Tuple([spaces.Discrete(6)] * self.num_robots)
        obs_dim = (self.num_robots * 2) + (self.num_boxes * 2) + (self.num_targets * 2) + 8
        self.observation_space = spaces.Box(
            low=-1, high=self.height + self.width,
            shape=(obs_dim,), dtype=np.float32
        )

    # ── Utilitários ──────────────────────────────────────────────────────────
    def _find_positions(self, symbols):
        if isinstance(symbols, str):
            symbols = [symbols]
        return [(i, j)
                for i in range(self.height)
                for j in range(self.width)
                if any(self.grid[i][j].startswith(s) for s in symbols)]

    def _remove_random_y_barriers(self):
        self.grid = [row[:] for row in self.base_grid]
        for (i, j) in self.y_positions:
            if random.random() < 0.5:
                self.grid[i][j] = '0'

    def reset(self):
        self._remove_random_y_barriers()
        self.robot_positions   = self._find_positions('R')
        self.box_positions     = self._find_positions('A')
        self.targets           = self._find_positions('B')
        self.delivered_boxes   = [False] * self.num_boxes
        self.steps             = 0
        self.total_deliveries  = 0
        self.collisions        = 0
        self.failures          = [0, 0]
        self.distance_traveled = [0, 0]
        self.previous_distances = [self._min_dist_to_boxes(r) for r in range(self.num_robots)]
        return self._get_obs(), self._get_info()

    def _min_dist_to_boxes(self, robot_id: int) -> float:
        rp = self.robot_positions[robot_id]
        boxes = [bp for bp_id, bp in enumerate(self.box_positions)
                 if bp is not None and not self.delivered_boxes[bp_id]]
        if not boxes:
            return 0.0
        return min(abs(rp[0] - bp[0]) + abs(rp[1] - bp[1]) for bp in boxes)

    def _is_valid(self, pos, robot_id=None) -> bool:
        i, j = pos
        if not (0 <= i < self.height and 0 <= j < self.width):
            return False
        if self.grid[i][j] in ('X', 'Y'):
            return False
        if robot_id is not None:
            for rid, rp in enumerate(self.robot_positions):
                if rid != robot_id and rp == (i, j):
                    return False
        return True

    def _alt_direction(self, action: int, robot_id: int):
        i, j = self.robot_positions[robot_id]
        alts = [a for a in range(4) if a != action]
        random.shuffle(alts)
        for a in alts:
            ni, nj = i, j
            if a == 0: ni -= 1
            elif a == 1: ni += 1
            elif a == 2: nj -= 1
            elif a == 3: nj += 1
            if self._is_valid((ni, nj), robot_id):
                return a, (ni, nj)
        return None, None

    # ── Mecânicas do ambiente ────────────────────────────────────────────────
    def _move(self, robot_id: int, action: int) -> float:
        i, j = self.robot_positions[robot_id]
        di = [-1, 1, 0, 0][action]
        dj = [0, 0, -1, 1][action]
        ni, nj = i + di, j + dj

        if random.random() < self.config.FAILURE_PROBABILITY:
            self.failures[robot_id] += 1
            if self.config.ALTERNATIVE_DIRECTIONS:
                _, alt_pos = self._alt_direction(action, robot_id)
                if alt_pos:
                    self.grid[i][j] = '0'
                    self.grid[alt_pos[0]][alt_pos[1]] = f'R{robot_id+1}'
                    self.robot_positions[robot_id] = alt_pos
                    self.distance_traveled[robot_id] += 1
            return self.config.FAILURE_PENALTY

        if self._is_valid((ni, nj), robot_id):
            self.grid[i][j] = '0'
            self.grid[ni][nj] = f'R{robot_id+1}'
            self.robot_positions[robot_id] = (ni, nj)
            self.distance_traveled[robot_id] += 1
            return -0.004
        else:
            self.collisions += 1
            return -0.02

    def _pickup(self, robot_id: int) -> float:
        rp = self.robot_positions[robot_id]
        for bid, bp in enumerate(self.box_positions):
            if not self.delivered_boxes[bid] and bp == rp:
                self.box_positions[bid] = None
                return 2.0
        return -0.02

    def _drop(self, robot_id: int) -> float:
        rp = self.robot_positions[robot_id]
        carrying = next((bid for bid, bp in enumerate(self.box_positions)
                         if bp is None and not self.delivered_boxes[bid]), None)
        if carrying is None:
            return -0.02
        if rp in self.targets:
            self.delivered_boxes[carrying] = True
            self.total_deliveries += 1
            return 30.0       # recompensa maior por entrega completa
        return -0.05

    def _shaped_reward(self, robot_id: int, base: float) -> float:
        r = base
        cur  = self._min_dist_to_boxes(robot_id)
        prev = self.previous_distances[robot_id]
        if cur < prev:
            r += 0.15 * (prev - cur)
        elif cur > prev:
            r -= 0.03 * (cur - prev)
        self.previous_distances[robot_id] = cur
        if all(self.delivered_boxes):
            r += 60.0        # bônus maior por completar todos
        return r

    # ── Observação ────────────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        obs = []
        for rp in self.robot_positions:
            obs += [rp[0] / self.height, rp[1] / self.width]
        for bid, bp in enumerate(self.box_positions):
            if bp is None or self.delivered_boxes[bid]:
                obs += [-1.0, -1.0]
            else:
                obs += [bp[0] / self.height, bp[1] / self.width]
        for tp in self.targets:
            obs += [tp[0] / self.height, tp[1] / self.width]
        for rp in self.robot_positions:
            boxes = [bp for bid, bp in enumerate(self.box_positions)
                     if bp is not None and not self.delivered_boxes[bid]]
            d_box = (min(abs(rp[0]-bp[0])+abs(rp[1]-bp[1]) for bp in boxes)
                     if boxes else 100)
            d_tgt = min(abs(rp[0]-tp[0])+abs(rp[1]-tp[1]) for tp in self.targets)
            obs += [d_box / (self.height+self.width), d_tgt / (self.height+self.width)]
        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict:
        return {
            'steps': self.steps,
            'total_deliveries': self.total_deliveries,
            'collisions': self.collisions,
            'failures_r1': self.failures[0],
            'failures_r2': self.failures[1],
            'distance_traveled': self.distance_traveled.copy(),
            'remaining_boxes': sum(1 for d in self.delivered_boxes if not d),
            'success_rate': self.total_deliveries / self.num_boxes,
        }

    # ── Step ─────────────────────────────────────────────────────────────────
    def step(self, actions):
        self.steps += 1
        rewards = []
        for rid, act in enumerate(actions):
            if act < 4:
                base = self._move(rid, act)
            elif act == 4:
                base = self._pickup(rid)
            else:
                base = self._drop(rid)
            rewards.append(self._shaped_reward(rid, base))

        terminated = all(self.delivered_boxes)
        truncated  = self.steps >= self.max_steps
        return self._get_obs(), rewards, terminated, truncated, self._get_info()

    def close(self):
        pass


# ==================== PLOTAGEM ====================
def _moving_avg(data, window=100):
    if len(data) < window:
        return None, None
    ma = np.convolve(data, np.ones(window) / window, mode='valid')
    x  = range(window, len(data) + 1)
    return x, ma


def _save_fig(fig, path, label):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✅ {label} salvo")


def plot_individual_graphs(metrics: dict, save_dir: Path):
    print(f"\n📊 Gerando gráficos individuais em: {save_dir}")
    n = len(metrics['episode_rewards'])
    ep = list(range(1, n + 1))

    # 1. Recompensa
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ep, metrics['episode_rewards'], color='steelblue', alpha=0.4, lw=1)
    x, ma = _moving_avg(metrics['episode_rewards'])
    if ma is not None:
        ax.plot(x, ma, 'r-', lw=2, label='Média Móvel (100 ep)')
    ax.set(xlabel='Episódio', ylabel='Recompensa Total',
           title='Recompensa por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_recompensa_vdn.png', 'grafico_recompensa_vdn.png')

    # 2. Entregas
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ep, metrics['episode_deliveries'], color='seagreen', alpha=0.4, lw=1)
    x, ma = _moving_avg(metrics['episode_deliveries'])
    if ma is not None:
        ax.plot(x, ma, 'r-', lw=2, label='Média Móvel (100 ep)')
    ax.axhline(y=8, color='orange', ls='--', lw=2, label='Meta (8 caixas)')
    ax.set(xlabel='Episódio', ylabel='Entregas', ylim=[-0.1, 8.5],
           title='Entregas por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_entregas_vdn.png', 'grafico_entregas_vdn.png')

    # 3. Steps
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ep, metrics['episode_steps'], color='darkorange', alpha=0.4, lw=1)
    x, ma = _moving_avg(metrics['episode_steps'])
    if ma is not None:
        ax.plot(x, ma, 'r-', lw=2, label='Média Móvel (100 ep)')
    ax.set(xlabel='Episódio', ylabel='Steps',
           title='Steps por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_steps_vdn.png', 'grafico_steps_vdn.png')

    # 4. Taxa de sucesso (acumulada por episódio)
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ep, metrics['success_rates'], color='mediumpurple', alpha=0.4, lw=1)
    x, ma = _moving_avg(metrics['success_rates'])
    if ma is not None:
        ax.plot(x, ma, 'r-', lw=2, label='Média Móvel (100 ep)')
    # Taxa de sucesso completo (8/8 entregas)
    ax.plot(ep, metrics['full_success'], color='gold', alpha=0.5, lw=1, label='Sucesso completo (8/8)')
    x2, ma2 = _moving_avg(metrics['full_success'])
    if ma2 is not None:
        ax.plot(x2, ma2, color='darkgoldenrod', lw=2, label='Média Móvel Sucesso Completo')
    ax.axhline(y=0.95, color='green', ls='--', lw=1.5, label='Meta 95%')
    ax.set(xlabel='Episódio', ylabel='Taxa de Sucesso', ylim=[-0.02, 1.08],
           title='Taxa de Sucesso por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_taxa_sucesso_vdn.png', 'grafico_taxa_sucesso_vdn.png')

    # 5. Colisões
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ep, metrics['collisions'], color='crimson', alpha=0.4, lw=1)
    x, ma = _moving_avg(metrics['collisions'])
    if ma is not None:
        ax.plot(x, ma, 'r-', lw=2, label='Média Móvel (100 ep)')
    ax.set(xlabel='Episódio', ylabel='Colisões',
           title='Colisões por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_colisoes_vdn.png', 'grafico_colisoes_vdn.png')

    # 6. Falhas por robô
    fig, ax = plt.subplots(figsize=(14, 7))
    total_f = np.array(metrics['failures_r1']) + np.array(metrics['failures_r2'])
    ax.plot(ep, total_f, color='saddlebrown', alpha=0.4, lw=1, label='Total')
    ax.plot(ep, metrics['failures_r1'], color='red',  alpha=0.35, lw=1, label='Robô 1')
    ax.plot(ep, metrics['failures_r2'], color='blue', alpha=0.35, lw=1, label='Robô 2')
    x, ma = _moving_avg(total_f.tolist())
    if ma is not None:
        ax.plot(x, ma, 'k-', lw=2, label='Média Móvel Total')
    ax.set(xlabel='Episódio', ylabel='Falhas',
           title='Falhas por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_falhas_vdn.png', 'grafico_falhas_vdn.png')

    # 7. Distância percorrida
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(ep, metrics['distance_traveled'], color='teal', alpha=0.4, lw=1)
    x, ma = _moving_avg(metrics['distance_traveled'])
    if ma is not None:
        ax.plot(x, ma, 'r-', lw=2, label='Média Móvel (100 ep)')
    ax.set(xlabel='Episódio', ylabel='Distância Percorrida',
           title='Distância Percorrida por Episódio — VDN (3000 episódios)')
    ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
    _save_fig(fig, save_dir / 'grafico_distancia_vdn.png', 'grafico_distancia_vdn.png')

    # 8. Loss de treinamento
    if metrics.get('losses'):
        fig, ax = plt.subplots(figsize=(14, 7))
        ls = metrics['losses']
        ax.plot(range(1, len(ls)+1), ls, color='slategray', alpha=0.4, lw=1)
        x, ma = _moving_avg(ls, window=min(200, len(ls)//5 or 1))
        if ma is not None:
            ax.plot(x, ma, 'r-', lw=2, label=f'Média Móvel')
        ax.set(xlabel='Step de treino', ylabel='Loss',
               title='Loss de Treinamento VDN (3000 episódios)')
        ax.legend(); ax.grid(True, alpha=0.3); fig.tight_layout()
        _save_fig(fig, save_dir / 'grafico_loss_vdn.png', 'grafico_loss_vdn.png')

    print(f"📊 Todos os gráficos salvos em: {save_dir}")


def save_metrics_csv(metrics: dict, save_dir: Path):
    df = pd.DataFrame({
        'episodio':           range(1, len(metrics['episode_rewards']) + 1),
        'recompensa':         metrics['episode_rewards'],
        'entregas':           metrics['episode_deliveries'],
        'steps':              metrics['episode_steps'],
        'taxa_sucesso':       metrics['success_rates'],
        'sucesso_completo':   metrics['full_success'],
        'colisoes':           metrics['collisions'],
        'falha_r1':           metrics['failures_r1'],
        'falha_r2':           metrics['failures_r2'],
        'distancia':          metrics['distance_traveled'],
    })
    path = save_dir / 'metricas.csv'
    df.to_csv(path, index=False)
    print(f"📊 Métricas salvas em: {path}")


# ==================== TREINAMENTO VDN ====================
def run_vdn_training(num_episodes: int = 3000):
    config = VDNConfig()
    config.EPISODES_TOTAL = num_episodes

    print("=" * 80)
    print("🤖 TREINAMENTO VDN — VALUE DECOMPOSITION NETWORKS")
    print("=" * 80)
    print(f"\n📋 CONFIGURAÇÃO:")
    print(f"   • Método:          VDN  (Q_total = Q1 + Q2)")
    print(f"   • Dispositivo:     {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"   • Episódios:       {num_episodes}")
    print(f"   • Max steps/ep:    {config.MAX_STEPS}")
    print(f"   • Batch size:      {config.BATCH_SIZE}")
    print(f"   • Learning rate:   {config.LEARNING_RATE}")
    print(f"   • Hidden dim:      {config.HIDDEN_DIM}")
    print(f"   • Gamma:           {config.GAMMA}")
    print(f"   • Falhas:          {config.FAILURE_PROBABILITY*100:.0f}%")
    print(f"   • Buffer:          {config.BUFFER_SIZE}")
    print("=" * 80)

    env = WarehouseEnv(config=config)
    obs, _ = env.reset()
    state_dim  = len(obs)
    action_dim = 6
    env.close()

    print(f"\n📊 DIMENSÕES: estado={state_dim}  ações={action_dim}  agentes=2  caixas=8")

    controller = VDNController(
        n_agents=2, state_dim=state_dim,
        action_dim=action_dim, config=config
    )

    metrics = {
        'episode_rewards':    [],
        'episode_deliveries': [],
        'episode_steps':      [],
        'success_rates':      [],
        'full_success':       [],
        'collisions':         [],
        'failures_r1':        [],
        'failures_r2':        [],
        'distance_traveled':  [],
        'losses':             [],
    }

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).resolve().parent.parent / "resultados" / "vdn_ajustado"
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Diretório de resultados: {results_dir.absolute()}")

    env = WarehouseEnv(config=config)
    t0  = time.time()

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0

        for step in range(config.MAX_STEPS):
            actions  = controller.select_actions(obs, training=True)
            nobs, rewards, terminated, truncated, info = env.step(actions)
            done     = terminated or truncated
            controller.remember(obs, actions, rewards, nobs, float(done))
            loss = controller.optimize()
            if loss:
                metrics['losses'].append(loss)
            ep_reward += sum(rewards)
            obs = nobs
            if done:
                break

        controller.scheduler.step()

        metrics['episode_rewards'].append(ep_reward)
        metrics['episode_deliveries'].append(info['total_deliveries'])
        metrics['episode_steps'].append(step + 1)
        metrics['success_rates'].append(info['success_rate'])
        metrics['full_success'].append(1.0 if info['total_deliveries'] == env.num_boxes else 0.0)
        metrics['collisions'].append(info['collisions'])
        metrics['failures_r1'].append(info['failures_r1'])
        metrics['failures_r2'].append(info['failures_r2'])
        metrics['distance_traveled'].append(sum(info['distance_traveled']))

        # Log a cada 100 episódios
        if (ep + 1) % 100 == 0:
            w   = 100
            rr  = metrics['episode_rewards'][-w:]
            rd  = metrics['episode_deliveries'][-w:]
            rfs = metrics['full_success'][-w:]
            eps = controller.get_epsilon()
            el  = (time.time() - t0) / 60
            print(f"Ep {ep+1:4d}/{num_episodes} | "
                  f"Reward: {np.mean(rr):8.2f} | "
                  f"Entregas: {np.mean(rd):.2f}/8 | "
                  f"Sucesso: {np.mean(rfs)*100:5.1f}% | "
                  f"ε: {eps:.3f} | "
                  f"Tempo: {el:.1f}min")

        if (ep + 1) % config.SAVE_INTERVAL == 0:
            controller.save(results_dir / "models", ep + 1)

        if (ep + 1) % config.CLEAN_MEMORY_EVERY == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    total_time = (time.time() - t0) / 60
    env.close()

    print(f"\n💾 SALVANDO RESULTADOS...")
    controller.save_final(results_dir / "models")
    save_metrics_csv(metrics, results_dir)
    plot_individual_graphs(metrics, results_dir)

    # Estatísticas finais
    w = 100
    fr = metrics['episode_rewards'][-w:]
    fd = metrics['episode_deliveries'][-w:]
    fs = metrics['full_success'][-w:]
    fc = metrics['collisions'][-w:]

    report = f"""
========================================
RELATÓRIO FINAL — VDN (3000 EPISÓDIOS)
========================================
DATA:      {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
DIRETÓRIO: {results_dir.absolute()}

CONFIGURAÇÃO:
  Método:          VDN (Value Decomposition Networks)
  Q_total          = Q1 + Q2  (decomposição aditiva)
  Episódios:       {num_episodes}
  Tempo total:     {total_time:.1f} min
  Batch Size:      {config.BATCH_SIZE}
  Learning Rate:   {config.LEARNING_RATE}
  Gamma:           {config.GAMMA}
  Hidden Dim:      {config.HIDDEN_DIM}
  Falhas robôs:    {config.FAILURE_PROBABILITY*100:.0f}%

MÉTRICAS (últimos 100 episódios):
  Recompensa média:       {np.mean(fr):.2f}
  Entregas médias:        {np.mean(fd):.2f}/8
  Taxa sucesso completo:  {np.mean(fs)*100:.1f}%  (8/8 caixas)
  Colisões médias:        {np.mean(fc):.1f}

MELHORES RESULTADOS:
  Melhor recompensa:  {max(metrics['episode_rewards']):.2f}
  Máx entregas/ep:    {max(metrics['episode_deliveries'])}/8
  Menor steps:        {min(metrics['episode_steps'])}

GRÁFICOS GERADOS (8):
  grafico_recompensa_vdn.png
  grafico_entregas_vdn.png
  grafico_steps_vdn.png
  grafico_taxa_sucesso_vdn.png
  grafico_colisoes_vdn.png
  grafico_falhas_vdn.png
  grafico_distancia_vdn.png
  grafico_loss_vdn.png
========================================
"""
    with open(results_dir / "relatorio.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    print(report)
    print(f"\n✅ TREINAMENTO VDN CONCLUÍDO!")
    print(f"📁 Resultados em: {results_dir.absolute()}")

    return controller, metrics, results_dir



# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser(description="VDN ajustado (standalone)")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--device", type=str, default=None, help="cpu | mps | cuda")
    parser.add_argument("--train-freq", type=int, default=None,
                        help="override VDNConfig.TRAIN_FREQ (default do notebook = 2)")
    args = parser.parse_args()
    if args.device:
        _os.environ["VDN_DEVICE"] = args.device
    if args.train_freq is not None:
        VDNConfig.TRAIN_FREQ = args.train_freq
    print("Dispositivo:", _pick_device(), "| TRAIN_FREQ:", VDNConfig.TRAIN_FREQ,
          "| MAX_STEPS:", VDNConfig.MAX_STEPS)
    t0 = time.time()
    controller, metrics, results_dir = run_vdn_training(num_episodes=args.episodes)
    dt = time.time() - t0
    n = len(metrics["episode_rewards"])
    print(f"\n[TIMING] {n} episodios em {dt/60:.2f} min "
          f"({dt/max(1,n):.3f} s/ep). Estimativa 3000 ep: {dt/max(1,n)*3000/60:.1f} min.")
