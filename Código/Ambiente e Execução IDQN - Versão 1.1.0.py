"""
IMPLEMENTAÇÃO COM MÚLTIPLAS SESSÕES DE TREINAMENTO CONTÍNUO
- Suporte para treinamento em múltiplas sessões de 1500 episódios cada
- Capacidade de carregar modelos anteriores e continuar treinamento
- Armazenamento completo de todas as condições e métricas
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
import pandas as pd
from datetime import datetime
import json
import os
from pathlib import Path
import imageio
from tqdm import tqdm
import pickle
import shutil
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIGURAÇÃO DO MAPA ====================
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

# ==================== CONFIGURAÇÃO OTIMIZADA ====================
class OptimizedConfig:
    """Configuração com todos os parâmetros otimizados"""
    
    # Exploração
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 50000
    
    # Aprendizado
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 256
    GAMMA = 0.95
    TAU = 0.001
    WEIGHT_DECAY = 1e-5
    
    # Rede Neural
    HIDDEN_DIM = 512
    NUM_LAYERS = 4
    DROPOUT_RATE = 0.2
    
    # Memória
    BUFFER_SIZE = 500000
    PRIORITIZED_REPLAY = True
    ALPHA = 0.6
    BETA = 0.4
    BETA_INCREMENT = 0.001
    
    # Regularização
    MAX_GRAD_NORM = 1.0
    
    # Treinamento
    LEARNING_STARTS = 1000
    TRAIN_FREQ = 4
    TARGET_UPDATE_FREQ = 100
    USE_SOFT_UPDATE = True
    
    # Ambiente
    MAX_STEPS = 500
    EPISODES_PER_SESSION = 1500  # Episódios por sessão
    
    # Diretórios
    BASE_DIR = "resultados_warehouse_multisession"

# ==================== PRIORITIZED REPLAY BUFFER ====================
class PrioritizedReplayBuffer:
    """Experience Replay Buffer com priorização - com suporte a save/load"""
    
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
            self.priorities[self.position] = max_priority
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:len(self.buffer)])
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)
    
    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
        
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        """Salva o buffer em disco"""
        data = {
            'capacity': self.capacity,
            'alpha': self.alpha,
            'beta': self.beta,
            'beta_increment': self.beta_increment,
            'buffer': self.buffer,
            'priorities': self.priorities,
            'position': self.position
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath):
        """Carrega o buffer do disco"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.capacity = data['capacity']
        self.alpha = data['alpha']
        self.beta = data['beta']
        self.beta_increment = data['beta_increment']
        self.buffer = data['buffer']
        self.priorities = data['priorities']
        self.position = data['position']

# ==================== REDE NEURAL ====================
class ImprovedDQN(nn.Module):
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
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

# ==================== AGENTE IDQN OTIMIZADO COM SUPORTE A CONTINUAÇÃO ====================
class OptimizedIDQNAgent:
    """Agente IDQN com suporte a save/load para treinamento contínuo"""
    
    def __init__(self, state_dim, action_dim, agent_id, config):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Redes
        self.policy_net = ImprovedDQN(state_dim, action_dim, 
                                      config.HIDDEN_DIM, 
                                      config.DROPOUT_RATE).to(self.device)
        self.target_net = ImprovedDQN(state_dim, action_dim,
                                      config.HIDDEN_DIM,
                                      config.DROPOUT_RATE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Otimizador
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                   lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
        
        # Memória
        if config.PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(
                config.BUFFER_SIZE,
                alpha=config.ALPHA,
                beta=config.BETA
            )
        else:
            self.memory = deque(maxlen=config.BUFFER_SIZE)
        
        # Contadores
        self.steps_done = 0
        self.learning_steps = 0
        self.total_episodes = 0
        
        # Métricas
        self.losses = []
        
    def get_epsilon(self):
        """Decaimento linear do epsilon baseado nos steps totais"""
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END
        
        epsilon = self.config.EPSILON_START - (self.config.EPSILON_START - self.config.EPSILON_END) * \
                  self.steps_done / self.config.EPSILON_DECAY_STEPS
        return max(self.config.EPSILON_END, epsilon)
    
    def select_action(self, state, training=True):
        self.steps_done += 1
        
        if training and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        if self.config.PRIORITIZED_REPLAY:
            self.memory.push(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def optimize(self):
        if len(self.memory) < self.config.BATCH_SIZE or \
           self.steps_done < self.config.LEARNING_STARTS:
            return 0
        
        self.learning_steps += 1
        
        if self.learning_steps % self.config.TRAIN_FREQ != 0:
            return 0
        
        if self.config.PRIORITIZED_REPLAY:
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.config.BATCH_SIZE)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.config.BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)
            indices = None
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)
        
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + self.config.GAMMA * next_q_values * (1 - dones)
        
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        td_errors = target_q - current_q
        
        if self.config.PRIORITIZED_REPLAY:
            loss = (weights * td_errors.pow(2)).mean()
        else:
            loss = td_errors.pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.MAX_GRAD_NORM)
        self.optimizer.step()
        
        if self.config.PRIORITIZED_REPLAY and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        
        if self.config.USE_SOFT_UPDATE:
            self.soft_update_target()
        elif self.learning_steps % self.config.TARGET_UPDATE_FREQ == 0:
            self.hard_update_target()
        
        self.losses.append(loss.item())
        
        return loss.item()
    
    def soft_update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(self.config.TAU * policy_param.data + 
                                   (1 - self.config.TAU) * target_param.data)
    
    def hard_update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_full_state(self, save_dir, session_id):
        """Salva estado completo do agente"""
        agent_dir = save_dir / f"agent_{self.agent_id}"
        agent_dir.mkdir(exist_ok=True)
        
        # Salvar modelos
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'learning_steps': self.learning_steps,
            'total_episodes': self.total_episodes,
            'losses': self.losses[-1000:] if self.losses else []
        }, agent_dir / f"model_session_{session_id}.pth")
        
        # Salvar memória
        if self.config.PRIORITIZED_REPLAY:
            self.memory.save(agent_dir / f"memory_session_{session_id}.pkl")
        else:
            with open(agent_dir / f"memory_session_{session_id}.pkl", 'wb') as f:
                pickle.dump(self.memory, f)
        
        return agent_dir
    
    def load_full_state(self, load_dir, session_id):
        """Carrega estado completo do agente"""
        agent_dir = load_dir / f"agent_{self.agent_id}"
        
        # Carregar modelos
        checkpoint = torch.load(agent_dir / f"model_session_{session_id}.pth")
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps_done = checkpoint['steps_done']
        self.learning_steps = checkpoint['learning_steps']
        self.total_episodes = checkpoint['total_episodes']
        self.losses = checkpoint['losses']
        
        # Carregar memória
        if self.config.PRIORITIZED_REPLAY:
            self.memory.load(agent_dir / f"memory_session_{session_id}.pkl")
        else:
            with open(agent_dir / f"memory_session_{session_id}.pkl", 'rb') as f:
                self.memory = pickle.load(f)

# ==================== AMBIENTE WAREHOUSE ====================
class WarehouseEnv(gym.Env):
    """Ambiente de Warehouse com 2 robôs movendo caixas"""
    
    metadata = {'render.modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode: str = None, seed: int = None, config=None):
        super().__init__()
        
        self.config = config or OptimizedConfig()
        self.height = MAP_CONFIG['height']
        self.width = MAP_CONFIG['width']
        self.grid = [row[:] for row in MAP_CONFIG['grid']]
        
        self.robot_positions = None
        self.box_positions = None
        self.targets = self._find_positions('B')
        self.obstacles = self._find_positions(['X', 'Y'])
        
        self.num_robots = 2
        self.num_boxes = len(self._find_positions('A'))
        self.num_targets = len(self.targets)
        
        self.delivered_boxes = None
        self.steps = 0
        self.max_steps = self.config.MAX_STEPS
        self.total_deliveries = 0
        self.collisions = 0
        self.distance_traveled = [0, 0]
        self.previous_distances = None
        
        self.action_space = spaces.Tuple([spaces.Discrete(6) for _ in range(self.num_robots)])
        
        obs_dim = (self.num_robots * 2) + (self.num_boxes * 2) + (self.num_targets * 2) + 8
        self.observation_space = spaces.Box(
            low=-1, high=self.height + self.width,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.render_mode = render_mode
        self.seed_value = seed
        if seed is not None:
            self.seed(seed)
        
        self.fig = None
        self.ax = None
        self.frame_buffer = []
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]
    
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
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        
        self.grid = [row[:] for row in MAP_CONFIG['grid']]
        self.robot_positions = self._find_positions('R')
        self.box_positions = self._find_positions('A')
        self.delivered_boxes = [False] * self.num_boxes
        
        self.steps = 0
        self.total_deliveries = 0
        self.collisions = 0
        self.distance_traveled = [0, 0]
        self.frame_buffer = []
        
        self.previous_distances = [self._min_distance_to_boxes(r) for r in range(self.num_robots)]
        
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, info
    
    def _min_distance_to_boxes(self, robot_id):
        robot_pos = self.robot_positions[robot_id]
        remaining_boxes = [box_pos for box_pos in self.box_positions 
                          if box_pos is not None and 
                          not self.delivered_boxes[self.box_positions.index(box_pos)]]
        
        if not remaining_boxes:
            return 0
        
        return min([self._manhattan_distance(robot_pos, box_pos) for box_pos in remaining_boxes])
    
    def _is_valid_position(self, pos, robot_id=None):
        i, j = pos
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return False
        
        cell = self.grid[i][j]
        if cell in ['X', 'Y']:
            return False
        
        if robot_id is not None:
            for rid, rpos in enumerate(self.robot_positions):
                if rid != robot_id and rpos == (i, j):
                    return False
        
        return True
    
    def _move_robot(self, robot_id, action):
        i, j = self.robot_positions[robot_id]
        new_i, new_j = i, j
        
        if action == 0: new_i -= 1
        elif action == 1: new_i += 1
        elif action == 2: new_j -= 1
        elif action == 3: new_j += 1
        
        if self._is_valid_position((new_i, new_j), robot_id):
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
            min_box_dist = min([self._manhattan_distance(robot_pos, box_pos) 
                               for box_pos in self.box_positions 
                               if box_pos is not None and 
                               not self.delivered_boxes[self.box_positions.index(box_pos)]],
                              default=100)
            obs.append(min_box_dist / (self.height + self.width))
            
            min_target_dist = min([self._manhattan_distance(robot_pos, target_pos) 
                                  for target_pos in self.targets],
                                 default=100)
            obs.append(min_target_dist / (self.height + self.width))
        
        return np.array(obs, dtype=np.float32)
    
    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def step(self, actions):
        self.steps += 1
        
        if len(actions) != self.num_robots:
            actions = [actions] * self.num_robots
        
        total_reward = 0
        rewards = [0, 0]
        
        movement_rewards = []
        for robot_id, action in enumerate(actions):
            if action < 4:
                reward = self._move_robot(robot_id, action)
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
        
        if self.render_mode in ['human', 'rgb_array']:
            frame = self.render()
            if frame is not None:
                self.frame_buffer.append(frame)
        
        return observation, rewards, terminated, truncated, info
    
    def _get_info(self):
        return {
            'steps': self.steps,
            'total_deliveries': self.total_deliveries,
            'collisions': self.collisions,
            'distance_traveled': self.distance_traveled.copy(),
            'remaining_boxes': sum(1 for d in self.delivered_boxes if not d),
            'success_rate': self.total_deliveries / self.num_boxes if self.steps > 0 else 0,
            'all_delivered': all(self.delivered_boxes)
        }
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_array()
        elif self.render_mode == 'human':
            self._render_human()
            return None
    
    def _render_array(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        self._draw_grid(ax)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img
    
    def _render_human(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.clear()
        self._draw_grid(self.ax)
        plt.pause(0.1)
    
    def _draw_grid(self, ax):
        colors = {
            '0': 'white', 'X': 'black', 'Y': 'gray',
            'R1': 'red', 'R2': 'blue', 'A': 'orange', 'B': 'green'
        }
        
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid[i][j]
                color = colors.get(cell, 'white')
                rect = Rectangle((j, self.height - 1 - i), 1, 1,
                               facecolor=color, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
                
                if cell in ['R1', 'R2', 'A', 'B']:
                    ax.text(j + 0.5, self.height - 0.5 - i, cell,
                           ha='center', va='center', fontweight='bold')
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Warehouse | Steps: {self.steps} | Entregas: {self.total_deliveries}/4')
    
    def get_frames(self):
        return self.frame_buffer.copy()
    
    def clear_frames(self):
        self.frame_buffer = []
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

# ==================== GERENCIADOR DE SESSÕES ====================
class SessionManager:
    """Gerencia múltiplas sessões de treinamento"""
    
    def __init__(self, base_dir="resultados_warehouse_multisession"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Arquivo de controle de sessões
        self.sessions_file = self.base_dir / "sessions_control.json"
        self.sessions = self._load_sessions()
        
    def _load_sessions(self):
        """Carrega informações das sessões existentes"""
        if self.sessions_file.exists():
            with open(self.sessions_file, 'r') as f:
                return json.load(f)
        return {'sessions': [], 'last_session_id': 0}
    
    def _save_sessions(self):
        """Salva informações das sessões"""
        with open(self.sessions_file, 'w') as f:
            json.dump(self.sessions, f, indent=2)
    
    def create_new_session(self, config):
        """Cria uma nova sessão de treinamento"""
        session_id = self.sessions['last_session_id'] + 1
        session_dir = self.base_dir / f"session_{session_id:04d}"
        session_dir.mkdir(exist_ok=True)
        
        session_info = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'episodes_completed': 0,
            'total_episodes_planned': config.EPISODES_PER_SESSION,
            'directory': str(session_dir),
            'config': {k: v for k, v in vars(config).items() if not k.startswith('_')}
        }
        
        self.sessions['sessions'].append(session_info)
        self.sessions['last_session_id'] = session_id
        self._save_sessions()
        
        return session_id, session_dir
    
    def get_last_session(self):
        """Retorna a última sessão existente"""
        if self.sessions['sessions']:
            last_session = self.sessions['sessions'][-1]
            return last_session['session_id'], Path(last_session['directory'])
        return None, None
    
    def update_session_progress(self, session_id, episodes_completed, metrics):
        """Atualiza o progresso de uma sessão"""
        for session in self.sessions['sessions']:
            if session['session_id'] == session_id:
                session['episodes_completed'] = episodes_completed
                session['last_update'] = datetime.now().isoformat()
                
                # Salvar métricas resumidas
                metrics_file = Path(session['directory']) / "session_summary.json"
                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)
                break
        
        self._save_sessions()
    
    def list_sessions(self):
        """Lista todas as sessões"""
        print("\n" + "=" * 60)
        print("📋 SESSÕES DE TREINAMENTO")
        print("=" * 60)
        for session in self.sessions['sessions']:
            print(f"\nSessão {session['session_id']:04d}:")
            print(f"   Data: {session['timestamp'][:19]}")
            print(f"   Episódios: {session['episodes_completed']}/{session['total_episodes_planned']}")
            print(f"   Diretório: {session['directory']}")
        print("=" * 60)

# ==================== SESSÃO DE TREINAMENTO ====================
class TrainingSession:
    """Representa uma sessão de treinamento"""
    
    def __init__(self, session_id, session_dir, config, load_from_previous=True):
        self.session_id = session_id
        self.session_dir = session_dir
        self.config = config
        self.load_from_previous = load_from_previous
        
        # Diretórios da sessão
        self.models_dir = session_dir / "models"
        self.metrics_dir = session_dir / "metrics"
        self.videos_dir = session_dir / "videos"
        self.checkpoints_dir = session_dir / "checkpoints"
        self.logs_dir = session_dir / "logs"
        
        for dir_path in [self.models_dir, self.metrics_dir, self.videos_dir, 
                        self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Inicializar métricas da sessão
        self.session_metrics = {
            'episode_rewards': [],
            'episode_deliveries': [],
            'episode_steps': [],
            'success_rates': [],
            'collisions': [],
            'distance_traveled': [],
            'losses': [],
            'episode_start': 0  # Número do episódio inicial
        }
        
        self.start_time = datetime.now()
        
    def save_session_config(self):
        """Salva configuração da sessão"""
        with open(self.session_dir / "session_config.json", 'w') as f:
            json.dump({
                'session_id': self.session_id,
                'timestamp': self.start_time.isoformat(),
                'config': {k: v for k, v in vars(self.config).items() if not k.startswith('_')},
                'load_from_previous': self.load_from_previous
            }, f, indent=2)
    
    def save_metrics(self):
        """Salva métricas da sessão"""
        df = pd.DataFrame({
            'episode': range(self.session_metrics['episode_start'], 
                           self.session_metrics['episode_start'] + len(self.session_metrics['episode_rewards'])),
            'reward': self.session_metrics['episode_rewards'],
            'deliveries': self.session_metrics['episode_deliveries'],
            'steps': self.session_metrics['episode_steps'],
            'success_rate': self.session_metrics['success_rates'],
            'collisions': self.session_metrics['collisions'],
            'distance': self.session_metrics['distance_traveled']
        })
        df.to_csv(self.metrics_dir / "training_metrics.csv", index=False)
        
        # Salvar também em formato pickle para carregamento rápido
        with open(self.metrics_dir / "training_metrics.pkl", 'wb') as f:
            pickle.dump(self.session_metrics, f)
    
    def save_checkpoint(self, agents, episode):
        """Salva checkpoint do treinamento"""
        checkpoint_dir = self.checkpoints_dir / f"checkpoint_{episode:06d}"
        checkpoint_dir.mkdir(exist_ok=True)
        
        for agent in agents:
            agent.save_full_state(checkpoint_dir, self.session_id)
        
        # Salvar métricas do checkpoint
        with open(checkpoint_dir / "metrics_checkpoint.pkl", 'wb') as f:
            pickle.dump({
                'episode': episode,
                'session_metrics': self.session_metrics,
                'timestamp': datetime.now().isoformat()
            }, f)
        
        # Remover checkpoints antigos (manter apenas últimos 5)
        checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_*"))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                shutil.rmtree(old_checkpoint)
    
    def load_checkpoint(self, agents, episode=None):
        """Carrega o checkpoint mais recente"""
        if episode is None:
            checkpoints = sorted(self.checkpoints_dir.glob("checkpoint_*"))
            if not checkpoints:
                return 0
            latest_checkpoint = checkpoints[-1]
        else:
            latest_checkpoint = self.checkpoints_dir / f"checkpoint_{episode:06d}"
            if not latest_checkpoint.exists():
                return 0
        
        # Carregar métricas
        with open(latest_checkpoint / "metrics_checkpoint.pkl", 'rb') as f:
            checkpoint_data = pickle.load(f)
            self.session_metrics = checkpoint_data['session_metrics']
            start_episode = checkpoint_data['episode']
        
        # Carregar agentes
        for agent in agents:
            agent.load_full_state(latest_checkpoint, self.session_id)
        
        return start_episode
    
    def log_message(self, message):
        """Registra mensagem no log da sessão"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self.logs_dir / "session.log", 'a', encoding='utf-8') as f:
            f.write(f"{timestamp} - {message}\n")
        print(message)
    
    def generate_session_report(self):
        """Gera relatório da sessão"""
        if not self.session_metrics['episode_rewards']:
            return
        
        rewards = self.session_metrics['episode_rewards']
        deliveries = self.session_metrics['episode_deliveries']
        success_rates = self.session_metrics['success_rates']
        
        report = f"""
        ========================================
        RELATÓRIO DA SESSÃO {self.session_id:04d}
        ========================================
        
        Duração: {datetime.now() - self.start_time}
        
        MÉTRICAS GERAIS:
        - Episódios: {len(rewards)}
        - Melhor recompensa: {max(rewards):.2f}
        - Recompensa média (últimos 100): {np.mean(rewards[-100:]):.2f}
        
        ENTREGAS:
        - Média entregas (últimos 100): {np.mean(deliveries[-100:]):.2f}
        - Máximo entregas: {max(deliveries)}
        - Taxa de sucesso final: {success_rates[-1]:.2%}
        
        EFICIÊNCIA:
        - Média steps: {np.mean(self.session_metrics['episode_steps'][-100:]):.1f}
        - Média colisões: {np.mean(self.session_metrics['collisions'][-100:]):.1f}
        
        ========================================
        """
        
        with open(self.session_dir / "session_report.txt", 'w') as f:
            f.write(report)
        
        self.log_message(report)

# ==================== TREINAMENTO PRINCIPAL ====================
def train_session(session, agents, resume=False):
    """Executa uma sessão de treinamento"""
    
    env = WarehouseEnv(config=session.config)
    
    # Carregar checkpoint se necessário
    start_episode = 0
    if resume:
        start_episode = session.load_checkpoint(agents)
        session.log_message(f"📂 Retomando treinamento do episódio {start_episode}")
    
    session.log_message(f"\n🚀 Iniciando sessão {session.session_id:04d}")
    session.log_message(f"   Episódios: {start_episode} → {start_episode + session.config.EPISODES_PER_SESSION}")
    
    total_episodes = start_episode + session.config.EPISODES_PER_SESSION
    session.session_metrics['episode_start'] = start_episode
    
    best_reward = -float('inf')
    save_checkpoint_every = 100  # Salvar checkpoint a cada 100 episódios
    
    for episode in tqdm(range(start_episode, total_episodes), 
                        desc=f"Sessão {session.session_id:04d}", 
                        initial=start_episode,
                        total=total_episodes):
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_collisions = 0
        episode_losses = []
        
        for step in range(session.config.MAX_STEPS):
            actions = [agent.select_action(obs) for agent in agents]
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            for i, agent in enumerate(agents):
                agent.remember(obs, actions[i], rewards[i], next_obs, terminated or truncated)
                episode_reward += rewards[i]
            
            episode_collisions = info['collisions']
            
            for agent in agents:
                loss = agent.optimize()
                if loss > 0:
                    episode_losses.append(loss)
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Atualizar total de episódios dos agentes
        for agent in agents:
            agent.total_episodes = episode + 1
        
        # Registrar métricas
        session.session_metrics['episode_rewards'].append(episode_reward)
        session.session_metrics['episode_deliveries'].append(info['total_deliveries'])
        session.session_metrics['episode_steps'].append(step + 1)
        session.session_metrics['success_rates'].append(info['success_rate'])
        session.session_metrics['collisions'].append(episode_collisions)
        session.session_metrics['distance_traveled'].append(sum(info['distance_traveled']))
        if episode_losses:
            session.session_metrics['losses'].extend(episode_losses)
        
        # Salvar melhor modelo
        if episode_reward > best_reward:
            best_reward = episode_reward
            for i, agent in enumerate(agents):
                agent.save_full_state(session.models_dir, f"best_ep{episode}")
        
        # Salvar checkpoint periódico
        if (episode + 1) % save_checkpoint_every == 0:
            session.save_checkpoint(agents, episode + 1)
            session.save_metrics()
        
        # Logging
        if (episode + 1) % 100 == 0:
            recent_rewards = session.session_metrics['episode_rewards'][-100:]
            recent_deliveries = session.session_metrics['episode_deliveries'][-100:]
            epsilon = agents[0].get_epsilon()
            
            log_msg = (f"Sessão {session.session_id:04d} | Ep {episode+1:5d} | "
                      f"Reward: {np.mean(recent_rewards):7.2f} | "
                      f"Entregas: {np.mean(recent_deliveries):.2f} | "
                      f"ε: {epsilon:.3f}")
            session.log_message(log_msg)
    
    env.close()
    
    # Salvar métricas finais da sessão
    session.save_metrics()
    session.generate_session_report()
    
    # Salvar modelos finais da sessão
    for i, agent in enumerate(agents):
        agent.save_full_state(session.models_dir, "final")
    
    return session.session_metrics

# ==================== FUNÇÃO PRINCIPAL ====================
def run_training_loop(num_sessions=4, resume_from_last=True):
    """Executa múltiplas sessões de treinamento contínuo"""
    
    config = OptimizedConfig()
    session_manager = SessionManager()
    
    print("=" * 80)
    print("🏭 TREINAMENTO MULTI-SESSÃO - WAREHOUSE COM 2 ROBÔS")
    print("=" * 80)
    print(f"\n📋 Configuração:")
    print(f"   • Episódios por sessão: {config.EPISODES_PER_SESSION}")
    print(f"   • Total de sessões: {num_sessions}")
    print(f"   • Total de episódios: {config.EPISODES_PER_SESSION * num_sessions}")
    print(f"   • Learning Rate: {config.LEARNING_RATE}")
    print(f"   • Batch Size: {config.BATCH_SIZE}")
    print(f"   • Epsilon Decay Steps: {config.EPSILON_DECAY_STEPS}")
    print("=" * 80)
    
    # Verificar se existe sessão anterior
    start_session = 1
    agents = None
    
    if resume_from_last:
        last_session_id, last_session_dir = session_manager.get_last_session()
        if last_session_dir is not None:
            print(f"\n📂 Sessão anterior encontrada: Sessão {last_session_id:04d}")
            response = input("Deseja continuar a partir da última sessão? (s/n): ")
            if response.lower() == 's':
                start_session = last_session_id + 1
                
                # Carregar agentes da última sessão
                print("Carregando agentes da última sessão...")
                env = WarehouseEnv(config=config)
                sample_obs, _ = env.reset()
                state_dim = len(sample_obs)
                action_dim = env.action_space[0].n
                env.close()
                
                agents = [OptimizedIDQNAgent(state_dim, action_dim, i, config) 
                         for i in range(2)]
                
                # Carregar o checkpoint final da última sessão
                last_session = TrainingSession(last_session_id, last_session_dir, config)
                last_session.load_checkpoint(agents)
                print(f"✓ Agentes carregados com {agents[0].total_episodes} episódios treinados")
    
    # Se não há agentes carregados, criar novos
    if agents is None:
        env = WarehouseEnv(config=config)
        sample_obs, _ = env.reset()
        state_dim = len(sample_obs)
        action_dim = env.action_space[0].n
        env.close()
        
        agents = [OptimizedIDQNAgent(state_dim, action_dim, i, config) 
                 for i in range(2)]
    
    # Executar sessões
    all_metrics = []
    
    for session_num in range(start_session, start_session + num_sessions):
        print(f"\n{'='*60}")
        print(f"🎯 INICIANDO SESSÃO {session_num:04d}")
        print(f"{'='*60}")
        
        # Criar nova sessão
        session_id, session_dir = session_manager.create_new_session(config)
        session = TrainingSession(session_id, session_dir, config, 
                                  load_from_previous=(session_num > start_session))
        session.save_session_config()
        
        # Executar treinamento da sessão
        session_metrics = train_session(session, agents, resume=(session_num > start_session))
        all_metrics.append(session_metrics)
        
        # Atualizar progresso no gerenciador
        total_episodes = agents[0].total_episodes
        session_manager.update_session_progress(session_id, total_episodes, {
            'final_reward_avg': np.mean(session_metrics['episode_rewards'][-100:]),
            'final_deliveries_avg': np.mean(session_metrics['episode_deliveries'][-100:]),
            'final_success_rate': session_metrics['success_rates'][-1]
        })
        
        print(f"\n✅ Sessão {session_id:04d} concluída!")
        print(f"   Total de episódios treinados: {total_episodes}")
    
    # Relatório final
    print("\n" + "=" * 80)
    print("🏆 TREINAMENTO MULTI-SESSÃO CONCLUÍDO!")
    print("=" * 80)
    
    # Listar todas as sessões
    session_manager.list_sessions()
    
    # Consolidar todas as métricas
    consolidated_metrics = {
        'episode_rewards': [],
        'episode_deliveries': [],
        'episode_steps': [],
        'success_rates': [],
        'collisions': [],
        'distance_traveled': []
    }
    
    for metrics in all_metrics:
        for key in consolidated_metrics:
            consolidated_metrics[key].extend(metrics[key])
    
    # Salvar métricas consolidadas
    consolidated_dir = Path(config.BASE_DIR) / "consolidated_results"
    consolidated_dir.mkdir(exist_ok=True)
    
    df_consolidated = pd.DataFrame({
        'episode': range(1, len(consolidated_metrics['episode_rewards']) + 1),
        'reward': consolidated_metrics['episode_rewards'],
        'deliveries': consolidated_metrics['episode_deliveries'],
        'steps': consolidated_metrics['episode_steps'],
        'success_rate': consolidated_metrics['success_rates'],
        'collisions': consolidated_metrics['collisions']
    })
    df_consolidated.to_csv(consolidated_dir / "consolidated_metrics.csv", index=False)
    
    # Plotar gráficos consolidados
    plot_consolidated_results(consolidated_metrics, consolidated_dir)
    
    # Salvar modelos finais consolidados
    final_models_dir = consolidated_dir / "final_models"
    final_models_dir.mkdir(exist_ok=True)
    for i, agent in enumerate(agents):
        agent.save_full_state(final_models_dir, "final")
    
    print(f"\n📁 Resultados consolidados salvos em: {consolidated_dir}")
    
    return agents, consolidated_metrics

def plot_consolidated_results(metrics, save_dir):
    """Plota gráficos consolidados de todas as sessões"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Recompensa consolidada
    axes[0, 0].plot(metrics['episode_rewards'], alpha=0.3, linewidth=0.5)
    if len(metrics['episode_rewards']) >= 100:
        moving_avg = np.convolve(metrics['episode_rewards'], np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(metrics['episode_rewards'])), moving_avg, 'r-', linewidth=2)
    axes[0, 0].set_title('Recompensa Total Consolidada')
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Recompensa')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Entregas consolidadas
    axes[0, 1].plot(metrics['episode_deliveries'], alpha=0.3, linewidth=0.5, color='green')
    if len(metrics['episode_deliveries']) >= 100:
        moving_avg_del = np.convolve(metrics['episode_deliveries'], np.ones(100)/100, mode='valid')
        axes[0, 1].plot(range(99, len(metrics['episode_deliveries'])), moving_avg_del, 'r-', linewidth=2)
    axes[0, 1].axhline(y=4, color='g', linestyle='--', label='Meta (4 entregas)')
    axes[0, 1].set_title('Entregas Consolidadas')
    axes[0, 1].set_xlabel('Episódio')
    axes[0, 1].set_ylabel('Entregas')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Steps consolidados
    axes[0, 2].plot(metrics['episode_steps'], color='orange', alpha=0.5)
    axes[0, 2].set_title('Steps por Episódio')
    axes[0, 2].set_xlabel('Episódio')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Taxa de sucesso consolidada
    axes[1, 0].plot(metrics['success_rates'], color='purple', alpha=0.5)
    if len(metrics['success_rates']) >= 100:
        moving_avg_success = np.convolve(metrics['success_rates'], np.ones(100)/100, mode='valid')
        axes[1, 0].plot(range(99, len(metrics['success_rates'])), moving_avg_success, 'r-', linewidth=2)
    axes[1, 0].axhline(y=0.95, color='g', linestyle='--', label='Meta 95%')
    axes[1, 0].set_title('Taxa de Sucesso Consolidada')
    axes[1, 0].set_xlabel('Episódio')
    axes[1, 0].set_ylabel('Taxa')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Colisões consolidadas
    axes[1, 1].plot(metrics['collisions'], color='red', alpha=0.5)
    if len(metrics['collisions']) >= 100:
        moving_avg_coll = np.convolve(metrics['collisions'], np.ones(100)/100, mode='valid')
        axes[1, 1].plot(range(99, len(metrics['collisions'])), moving_avg_coll, 'r-', linewidth=2)
    axes[1, 1].set_title('Colisões Consolidadas')
    axes[1, 1].set_xlabel('Episódio')
    axes[1, 1].set_ylabel('Colisões')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Resumo final
    axes[1, 2].axis('off')
    final_stats = {
        'Total Episódios': len(metrics['episode_rewards']),
        'Melhor Recompensa': f"{np.max(metrics['episode_rewards']):.2f}",
        'Média Entregas (últimos 100)': f"{np.mean(metrics['episode_deliveries'][-100:]):.2f}",
        'Taxa Sucesso Final': f"{metrics['success_rates'][-1]:.1%}",
        'Média Colisões (últimos 100)': f"{np.mean(metrics['collisions'][-100:]):.1f}"
    }
    
    axes[1, 2].text(0.1, 0.9, "📊 RESUMO FINAL", fontsize=12, fontweight='bold')
    y = 0.8
    for key, value in final_stats.items():
        axes[1, 2].text(0.1, y, f"{key}: {value}", fontsize=10)
        y -= 0.1
    
    plt.tight_layout()
    plt.savefig(save_dir / "consolidated_results.png", dpi=150)
    plt.close()

# ==================== EXECUÇÃO PRINCIPAL ====================
if __name__ == "__main__":
    # Número de sessões desejado (cada sessão = 1500 episódios)
    NUM_SESSIONS = 4  # Total = 6000 episódios
    
    # Executar treinamento multi-sessão
    agents, consolidated_metrics = run_training_loop(
        num_sessions=NUM_SESSIONS,
        resume_from_last=True  # Continua da última sessão automaticamente
    )
    
    print("\n✨ TREINAMENTO MULTI-SESSÃO CONCLUÍDO COM SUCESSO! ✨")
    print(f"📁 Todos os resultados estão em: resultados_warehouse_multisession/")