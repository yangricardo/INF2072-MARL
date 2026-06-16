"""Buffer de repetição com priorização (Prioritized Experience Replay)."""

import numpy as np


class PrioritizedReplayBuffer:
    """Replay buffer com amostragem proporcional ao erro TD (PER)."""

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
            priorities = np.array(self.priorities[: len(self.buffer)])

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)


class VDNPrioritizedReplayBuffer:
    """PER de transições conjuntas (estado, ações[], recompensas[], ...) para o VDN.

    Extraído de: Código/Executa Experimento VDN.ipynb (classe PrioritizedReplayBuffer).
    """

    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: list = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
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
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            indices,
            weights.astype(np.float32),
        )

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p_raw = abs(err) + 1e-6  # store raw value (without alpha)
            self.priorities[idx] = p_raw ** self.alpha  # apply alpha only when storing
            if p_raw > self._max_priority:
                self._max_priority = p_raw  # store raw max priority

    def __len__(self):
        return len(self.buffer)


class QMIXPrioritizedReplayBuffer:
    """PER conjunto que também guarda o estado global (atual e próximo) para o QMIX.

    Extraído de: Código/Executa - Experimento.ipynb (classe QMIXPrioritizedReplayBuffer).
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = []
        self.position = 0

    def push(self, state, actions, rewards, next_state, done, global_state, next_global_state):
        max_priority = max(self.priorities) if self.priorities else 1.0
        transition = (state, actions, rewards, next_state, done, global_state, next_global_state)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(max_priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[: len(self.buffer)])

        probs = priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
        (states, actions, rewards, next_states, dones,
         global_states, next_global_states) = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            np.array(global_states),
            np.array(next_global_states),
            indices,
            weights,
        )

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha

    def __len__(self):
        return len(self.buffer)
