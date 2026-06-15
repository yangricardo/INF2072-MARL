"""
IMPLEMENTAÇÃO CORRIGIDA PARA GOOGLE COLAB
- Tratamento de erros de sistema de arquivos
- Salvamento mais robusto
- Checkpoints opcionais
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
import warnings
import time
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

class OptimizedConfig:
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 50000
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 256
    GAMMA = 0.95
    TAU = 0.001
    WEIGHT_DECAY = 1e-5
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.2
    BUFFER_SIZE = 500000
    PRIORITIZED_REPLAY = True
    ALPHA = 0.6
    BETA = 0.4
    MAX_GRAD_NORM = 1.0
    LEARNING_STARTS = 1000
    TRAIN_FREQ = 4
    USE_SOFT_UPDATE = True
    MAX_STEPS = 500
    EPISODES_PER_SESSION = 1500
    BASE_DIR = "resultados_warehouse_colab"
    SAVE_CHECKPOINTS = True  # Desativar se houver problemas
    SAVE_CHECKPOINT_EVERY = 500  # Salvar com menos frequência
    COMPRESS_CHECKPOINTS = False

# ==================== PRIORITIZED REPLAY BUFFER ====================
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
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
    
    def __len__(self):
        return len(self.buffer)

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
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

# ==================== AGENTE IDQN ====================
class OptimizedIDQNAgent:
    def __init__(self, state_dim, action_dim, agent_id, config):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = ImprovedDQN(state_dim, action_dim, 
                                      config.HIDDEN_DIM, 
                                      config.DROPOUT_RATE).to(self.device)
        self.target_net = ImprovedDQN(state_dim, action_dim,
                                      config.HIDDEN_DIM,
                                      config.DROPOUT_RATE).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                   lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
        
        if config.PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.ALPHA)
        else:
            self.memory = deque(maxlen=config.BUFFER_SIZE)
        
        self.steps_done = 0
        self.learning_steps = 0
        self.total_episodes = 0
        self.losses = []
        
    def get_epsilon(self):
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
        
        self.losses.append(loss.item())
        return loss.item()
    
    def soft_update_target(self):
        for target_param, policy_param in zip(self.target_net.parameters(),
                                               self.policy_net.parameters()):
            target_param.data.copy_(self.config.TAU * policy_param.data + 
                                   (1 - self.config.TAU) * target_param.data)
    
    def save_checkpoint(self, save_dir):
        """Salva checkpoint de forma segura"""
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save({
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'steps_done': self.steps_done,
                'learning_steps': self.learning_steps,
                'total_episodes': self.total_episodes,
            }, save_dir / f"agent_{self.agent_id}.pth")
            return True
        except Exception as e:
            print(f"  ⚠️ Erro ao salvar checkpoint do agente {self.agent_id}: {e}")
            return False
    
    def load_checkpoint(self, load_dir, agent_id):
        """Carrega checkpoint de forma segura"""
        try:
            checkpoint_path = load_dir / f"agent_{agent_id}.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.target_net.load_state_dict(checkpoint['target_net'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.steps_done = checkpoint['steps_done']
                self.learning_steps = checkpoint['learning_steps']
                self.total_episodes = checkpoint['total_episodes']
                return True
        except Exception as e:
            print(f"  ⚠️ Erro ao carregar checkpoint do agente {agent_id}: {e}")
        return False

# ==================== AMBIENTE WAREHOUSE ====================
class WarehouseEnv(gym.Env):
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
        
        return observation, info
    
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
        return None
    
    def _render_array(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        self._draw_grid(ax)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img
    
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
    
    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

# ==================== TREINAMENTO ====================
def train_session(session_dir, agents, config, session_id=1, start_episode=0):
    """Executa uma sessão de treinamento com tratamento de erros robusto"""
    
    env = WarehouseEnv(config=config)
    total_episodes = start_episode + config.EPISODES_PER_SESSION
    
    # Criar diretórios com tratamento de erro
    try:
        os.makedirs(session_dir, exist_ok=True)
        metrics_dir = session_dir / "metrics"
        models_dir = session_dir / "models"
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Erro ao criar diretórios: {e}")
        # Usar diretório local alternativo
        session_dir = Path("./temp_training")
        metrics_dir = session_dir / "metrics"
        models_dir = session_dir / "models"
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    
    # Métricas
    metrics = {
        'episode_rewards': [],
        'episode_deliveries': [],
        'episode_steps': [],
        'success_rates': [],
        'collisions': [],
        'distance_traveled': []
    }
    
    best_reward = -float('inf')
    
    print(f"\n🚀 Iniciando sessão {session_id:04d}")
    print(f"   Episódios: {start_episode} → {total_episodes}")
    
    for episode in tqdm(range(start_episode, total_episodes), 
                        desc=f"Sessão {session_id:04d}",
                        total=total_episodes - start_episode):
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_collisions = 0
        
        for step in range(config.MAX_STEPS):
            actions = [agent.select_action(obs) for agent in agents]
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            
            for i, agent in enumerate(agents):
                agent.remember(obs, actions[i], rewards[i], next_obs, terminated or truncated)
                episode_reward += rewards[i]
            
            episode_collisions = info['collisions']
            
            for agent in agents:
                agent.optimize()
            
            obs = next_obs
            
            if terminated or truncated:
                break
        
        # Atualizar total de episódios
        for agent in agents:
            agent.total_episodes = episode + 1
        
        # Registrar métricas
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_deliveries'].append(info['total_deliveries'])
        metrics['episode_steps'].append(step + 1)
        metrics['success_rates'].append(info['success_rate'])
        metrics['collisions'].append(episode_collisions)
        metrics['distance_traveled'].append(sum(info['distance_traveled']))
        
        # Salvar melhor modelo
        if episode_reward > best_reward:
            best_reward = episode_reward
            for i, agent in enumerate(agents):
                try:
                    torch.save(agent.policy_net.state_dict(), 
                              models_dir / f"best_agent_{i}_ep{episode}.pth")
                except Exception as e:
                    pass  # Ignorar erro de salvamento
        
        # Salvar checkpoint periódico (menos frequente)
        if config.SAVE_CHECKPOINTS and (episode + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            try:
                checkpoint_dir = session_dir / f"checkpoint_{episode+1}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                for i, agent in enumerate(agents):
                    agent.save_checkpoint(checkpoint_dir)
                
                # Salvar métricas
                pd.DataFrame({
                    'episode': range(len(metrics['episode_rewards'])),
                    'reward': metrics['episode_rewards'],
                    'deliveries': metrics['episode_deliveries']
                }).to_csv(metrics_dir / f"metrics_checkpoint_{episode+1}.csv", index=False)
                
            except Exception as e:
                print(f"\n  ⚠️ Erro ao salvar checkpoint: {e}")
        
        # Logging
        if (episode + 1) % 100 == 0:
            recent_rewards = metrics['episode_rewards'][-100:]
            recent_deliveries = metrics['episode_deliveries'][-100:]
            epsilon = agents[0].get_epsilon()
            
            log_msg = (f"Sessão {session_id:04d} | Ep {episode+1:5d} | "
                      f"Reward: {np.mean(recent_rewards):7.2f} | "
                      f"Entregas: {np.mean(recent_deliveries):.2f} | "
                      f"ε: {epsilon:.3f}")
            print(log_msg)
    
    env.close()
    
    # Salvar métricas finais
    try:
        df = pd.DataFrame({
            'episode': range(start_episode, len(metrics['episode_rewards']) + start_episode),
            'reward': metrics['episode_rewards'],
            'deliveries': metrics['episode_deliveries'],
            'steps': metrics['episode_steps'],
            'success_rate': metrics['success_rates'],
            'collisions': metrics['collisions']
        })
        df.to_csv(metrics_dir / "training_metrics.csv", index=False)
    except Exception as e:
        print(f"⚠️ Erro ao salvar métricas: {e}")
    
    return metrics

# ==================== FUNÇÃO PRINCIPAL ====================
def run_training_loop(num_sessions=4):
    """Executa múltiplas sessões de treinamento"""
    
    config = OptimizedConfig()
    
    print("=" * 80)
    print("🏭 TREINAMENTO MULTI-SESSÃO - WAREHOUSE COM 2 ROBÔS")
    print("=" * 80)
    print(f"\n📋 Configuração:")
    print(f"   • Episódios por sessão: {config.EPISODES_PER_SESSION}")
    print(f"   • Total de sessões: {num_sessions}")
    print(f"   • Total de episódios: {config.EPISODES_PER_SESSION * num_sessions}")
    print(f"   • Checkpoints a cada: {config.SAVE_CHECKPOINT_EVERY} episódios")
    print("=" * 80)
    
    # Criar diretório base
    base_dir = Path(config.BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)
    
    # Inicializar ambiente para obter dimensões
    env = WarehouseEnv(config=config)
    sample_obs, _ = env.reset()
    state_dim = len(sample_obs)
    action_dim = env.action_space[0].n
    env.close()
    
    # Criar agentes
    agents = [OptimizedIDQNAgent(state_dim, action_dim, i, config) 
              for i in range(2)]
    
    all_metrics = []
    total_episodes_done = 0
    
    # Verificar se existe sessão anterior para continuar
    last_session_dir = None
    for session_num in range(1, num_sessions + 1):
        session_dir = base_dir / f"session_{session_num:04d}"
        
        # Tentar carregar checkpoint da sessão anterior
        if session_num > 1 and last_session_dir:
            checkpoint_dir = last_session_dir / f"checkpoint_{config.EPISODES_PER_SESSION}"
            if checkpoint_dir.exists():
                print(f"\n📂 Carregando checkpoint da sessão anterior...")
                for i, agent in enumerate(agents):
                    agent.load_checkpoint(checkpoint_dir, i)
        
        # Executar sessão
        session_metrics = train_session(
            session_dir, 
            agents, 
            config, 
            session_id=session_num,
            start_episode=total_episodes_done
        )
        
        all_metrics.append(session_metrics)
        total_episodes_done += config.EPISODES_PER_SESSION
        last_session_dir = session_dir
        
        # Salvar modelos da sessão
        try:
            models_dir = session_dir / "models"
            for i, agent in enumerate(agents):
                torch.save(agent.policy_net.state_dict(), 
                          models_dir / f"agent_{i}_final.pth")
        except Exception as e:
            print(f"⚠️ Erro ao salvar modelo final: {e}")
        
        print(f"\n✅ Sessão {session_num:04d} concluída!")
        print(f"   Total de episódios treinados: {total_episodes_done}")
    
    # Consolidar resultados
    consolidated_metrics = {
        'episode_rewards': [],
        'episode_deliveries': [],
        'episode_steps': [],
        'success_rates': [],
        'collisions': []
    }
    
    episode_offset = 0
    for session_num, metrics in enumerate(all_metrics, 1):
        for key in consolidated_metrics:
            consolidated_metrics[key].extend(metrics[key])
    
    # Salvar resultados consolidados
    consolidated_dir = base_dir / "consolidated_results"
    os.makedirs(consolidated_dir, exist_ok=True)
    
    df_consolidated = pd.DataFrame({
        'episode': range(1, len(consolidated_metrics['episode_rewards']) + 1),
        'reward': consolidated_metrics['episode_rewards'],
        'deliveries': consolidated_metrics['episode_deliveries'],
        'steps': consolidated_metrics['episode_steps'],
        'success_rate': consolidated_metrics['success_rates'],
        'collisions': consolidated_metrics['collisions']
    })
    df_consolidated.to_csv(consolidated_dir / "consolidated_metrics.csv", index=False)
    
    # Plotar gráficos
    plot_consolidated_results(consolidated_metrics, consolidated_dir)
    
    print(f"\n📁 Resultados consolidados salvos em: {consolidated_dir}")
    
    return agents, consolidated_metrics

def plot_consolidated_results(metrics, save_dir):
    """Plota gráficos consolidados"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Recompensa
    axes[0, 0].plot(metrics['episode_rewards'], alpha=0.3, linewidth=0.5)
    if len(metrics['episode_rewards']) >= 100:
        moving_avg = np.convolve(metrics['episode_rewards'], np.ones(100)/100, mode='valid')
        axes[0, 0].plot(range(99, len(metrics['episode_rewards'])), moving_avg, 'r-', linewidth=2)
    axes[0, 0].set_title('Recompensa Total Consolidada')
    axes[0, 0].set_xlabel('Episódio')
    axes[0, 0].set_ylabel('Recompensa')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Entregas
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
    
    # 3. Steps
    axes[0, 2].plot(metrics['episode_steps'], color='orange', alpha=0.5)
    axes[0, 2].set_title('Steps por Episódio')
    axes[0, 2].set_xlabel('Episódio')
    axes[0, 2].set_ylabel('Steps')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Taxa de sucesso
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
    
    # 5. Colisões
    axes[1, 1].plot(metrics['collisions'], color='red', alpha=0.5)
    if len(metrics['collisions']) >= 100:
        moving_avg_coll = np.convolve(metrics['collisions'], np.ones(100)/100, mode='valid')
        axes[1, 1].plot(range(99, len(metrics['collisions'])), moving_avg_coll, 'r-', linewidth=2)
    axes[1, 1].set_title('Colisões Consolidadas')
    axes[1, 1].set_xlabel('Episódio')
    axes[1, 1].set_ylabel('Colisões')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Resumo
    axes[1, 2].axis('off')
    final_stats = {
        'Total Episódios': len(metrics['episode_rewards']),
        'Melhor Recompensa': f"{np.max(metrics['episode_rewards']):.2f}",
        'Média Entregas': f"{np.mean(metrics['episode_deliveries'][-100:]):.2f}",
        'Taxa Sucesso Final': f"{metrics['success_rates'][-1]:.1%}"
    }
    
    axes[1, 2].text(0.1, 0.9, "📊 RESUMO FINAL", fontsize=12, fontweight='bold')
    y = 0.8
    for key, value in final_stats.items():
        axes[1, 2].text(0.1, y, f"{key}: {value}", fontsize=10)
        y -= 0.1
    
    plt.tight_layout()
    plt.savefig(save_dir / "consolidated_results.png", dpi=150)
    plt.close()

# ==================== EXECUÇÃO ====================
if __name__ == "__main__":
    # Desativar checkpoints frequentes para evitar erros
    OptimizedConfig.SAVE_CHECKPOINTS = True
    OptimizedConfig.SAVE_CHECKPOINT_EVERY = 500  # Salvar a cada 500 episódios
    
    # Número de sessões (cada sessão = 1500 episódios)
    NUM_SESSIONS = 2  # Começar com 2 sessões para teste
    
    try:
        agents, consolidated_metrics = run_training_loop(num_sessions=NUM_SESSIONS)
        print("\n✨ TREINAMENTO CONCLUÍDO COM SUCESSO! ✨")
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        print("\nTentando salvar checkpoint de emergência...")
        # Salvar modelos mesmo em caso de erro
        try:
            emergency_dir = Path("emergency_checkpoint")
            os.makedirs(emergency_dir, exist_ok=True)
            for i, agent in enumerate(agents):
                torch.save(agent.policy_net.state_dict(), 
                          emergency_dir / f"emergency_agent_{i}.pth")
            print(f"✓ Checkpoint de emergência salvo em {emergency_dir}")
        except:
            pass