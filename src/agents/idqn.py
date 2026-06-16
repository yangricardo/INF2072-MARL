"""Agente IDQN (Independent Deep Q-Network).

Extraído de: Código/Ambiente e Execução IDQN - Versão 1.3.0.py
(classe OptimizedIDQNAgent). Double-DQN com prioritized replay, soft update da
target network e exploração epsilon-greedy decrescente.

O fallback sem priorização (``deque``) é mantido por fidelidade ao original, ainda
que nenhuma config do projeto desative ``PRIORITIZED_REPLAY``.
"""

import os
import random
from collections import deque

import numpy as np
import torch
import torch.optim as optim

from ..networks import ImprovedDQN
from ..replay_buffer import PrioritizedReplayBuffer


class IDQNAgent:
    def __init__(self, state_dim, action_dim, agent_id, config):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cpu")

        self.policy_net = ImprovedDQN(
            state_dim, action_dim, config.HIDDEN_DIM, config.DROPOUT_RATE
        ).to(self.device)
        self.target_net = ImprovedDQN(
            state_dim, action_dim, config.HIDDEN_DIM, config.DROPOUT_RATE
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        self.memory: PrioritizedReplayBuffer | deque
        if config.PRIORITIZED_REPLAY:
            self.memory = PrioritizedReplayBuffer(
                config.BUFFER_SIZE,
                alpha=config.ALPHA,
                beta_start=config.BETA_START,
                beta_end=config.BETA_END,
            )
        else:
            self.memory = deque(maxlen=config.BUFFER_SIZE)

        self.steps_done = 0
        self.learning_steps = 0
        self.total_episodes = 0
        self.losses = []
        self._total_train_steps: int | None = None

    def set_total_train_steps(self, total_steps: int):
        """Define o total de steps de treino para o annealing do beta no PER."""
        self._total_train_steps = total_steps
        if self.config.PRIORITIZED_REPLAY and isinstance(self.memory, PrioritizedReplayBuffer):
            self.memory.set_total_steps(total_steps)

    def get_epsilon(self):
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END

        # ε-greedy linear annealing: ε(t) = ε_start - (ε_start - ε_end) · t / T
        epsilon = self.config.EPSILON_START - (
            self.config.EPSILON_START - self.config.EPSILON_END
        ) * self.steps_done / self.config.EPSILON_DECAY_STEPS
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
            self.memory.push(state, action, reward, next_state, done)  # type: ignore[union-attr]
        else:
            self.memory.append((state, action, reward, next_state, done))  # type: ignore[union-attr]

    def optimize(self):
        if (
            len(self.memory) < self.config.BATCH_SIZE
            or self.steps_done < self.config.LEARNING_STARTS
        ):
            return 0

        self.learning_steps += 1

        if self.learning_steps % self.config.TRAIN_FREQ != 0:
            return 0

        weights = None
        indices = None
        if self.config.PRIORITIZED_REPLAY:
            (
                states,
                actions,
                rewards,
                next_states,
                dones,
                indices,
                weights,
            ) = self.memory.sample(self.config.BATCH_SIZE, steps_done=self.steps_done)  # type: ignore[union-attr]
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            batch = random.sample(self.memory, self.config.BATCH_SIZE)  # type: ignore[arg-type]
            states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).to(self.device)

        with torch.no_grad():
            # Double-DQN: y = r + γ·Q_target(s', argmax_a' Q_policy(s',a'))·(1-done)
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q = rewards + self.config.GAMMA * next_q_values * (1 - dones)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD error: δ = y - Q(s,a)
        td_errors = target_q - current_q

        # PER loss: L = E_i[w_i · δ_i²] where w_i = (N·P(i))^(-β) / max(w)
        if self.config.PRIORITIZED_REPLAY and weights is not None:
            loss = (weights * td_errors.pow(2)).mean()
        else:
            loss = td_errors.pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.config.MAX_GRAD_NORM
        )
        self.optimizer.step()

        if self.config.PRIORITIZED_REPLAY and indices is not None:
            priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)  # type: ignore[union-attr]

        if self.config.USE_SOFT_UPDATE:
            self.soft_update_target()

        self.losses.append(loss.item())
        return loss.item()

    def soft_update_target(self):
        # Polyak averaging (soft update): θ_target ← τ·θ + (1-τ)·θ_target
        # Usa _foreach_lerp_ para vetorizar a operação sobre todos os parâmetros
        torch._foreach_lerp_(
            list(self.target_net.parameters()),
            list(self.policy_net.parameters()),
            self.config.TAU,
        )

    def save_checkpoint(self, save_dir):
        try:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(
                {
                    "policy_net": self.policy_net.state_dict(),
                    "target_net": self.target_net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "steps_done": self.steps_done,
                    "learning_steps": self.learning_steps,
                    "total_episodes": self.total_episodes,
                },
                save_dir / f"agent_{self.agent_id}.pth",
            )
            return True
        except Exception as e:
            print(f"  ⚠️ Erro ao salvar checkpoint do agente {self.agent_id}: {e}")
            return False

    def load_checkpoint(self, load_dir, agent_id):
        try:
            checkpoint_path = load_dir / f"agent_{agent_id}.pth"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint["policy_net"])
                self.target_net.load_state_dict(checkpoint["target_net"])
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                self.steps_done = checkpoint["steps_done"]
                self.learning_steps = checkpoint["learning_steps"]
                self.total_episodes = checkpoint["total_episodes"]
                return True
        except Exception as e:
            print(f"  ⚠️ Erro ao carregar checkpoint do agente {agent_id}: {e}")
        return False
