"""Buffer de repetição com priorização (Prioritized Experience Replay).

Implementa SumTree (Segment Tree) para amostragem O(log N) em vez de O(N),
e beta annealing para correção de viés das importance-sampling weights.
"""

import numpy as np


class SumTree:
    """Árvore binária de soma vetorizada de alta performance.
    Elimina loops Python usando indexação avançada do NumPy.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.sum_tree = np.zeros(2 * capacity, dtype=np.float64)
        self.max_tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write_index = 0
        self._size = 0

    def total(self) -> float:
        return float(self.sum_tree[1])

    def max_priority(self) -> float:
        if self._size == 0:
            return 1.0
        return float(self.max_tree[1])

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def add(self, priority: float):
        """Insere um elemento isolado (usado no push)."""
        idx = self.capacity + self._write_index
        self.sum_tree[idx] = priority
        self.max_tree[idx] = priority
        
        # Propagação clássica para um único item (leve)
        parent = idx // 2
        while parent >= 1:
            left = 2 * parent
            right = left + 1
            self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
            self.max_tree[parent] = max(self.max_tree[left], self.max_tree[right])
            parent //= 2

        self._write_index = (self._write_index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Amostra um lote inteiro de uma só vez sem nenhum laço 'for' em Python."""
        total = self.total()
        if total == 0.0:
            return rng.integers(0, self._size, size=batch_size)

        # Amostragem estratificada puramente vetorizada
        segment = total / batch_size
        a = segment * np.arange(batch_size)
        b = segment * (np.arange(batch_size) + 1)
        values = rng.uniform(a, b)

        # Desce na árvore com o lote inteiro simultaneamente
        idx = np.ones(batch_size, dtype=np.int64)
        while np.any(idx < self.capacity):
            left = 2 * idx
            right = left + 1
            
            # Captura a árvore de somas dos filhos esquerdos de todo o batch
            left_sums = self.sum_tree[left]
            
            # Mascaramento Booleano Vetorizado
            go_left = values <= left_sums
            idx = np.where(go_left, left, right)
            values = np.where(go_left, values, values - left_sums)

        return idx - self.capacity

    def batch_update(self, indices: np.ndarray, priorities: np.ndarray):
        """Atualiza múltiplos índices de uma vez só, subindo a árvore por camadas."""
        idx = self.capacity + indices
        self.sum_tree[idx] = priorities
        self.max_tree[idx] = priorities

        # Sobe a árvore camada por camada (profundidade máxima fixa log2 de capacity)
        parents = idx // 2
        while np.any(parents >= 1):
            unique_parents = np.unique(parents)
            if unique_parents[0] == 0:
                unique_parents = unique_parents[unique_parents >= 1]
            
            left = 2 * unique_parents
            right = left + 1
            
            self.sum_tree[unique_parents] = self.sum_tree[left] + self.sum_tree[right]
            self.max_tree[unique_parents] = np.maximum(self.max_tree[left], self.max_tree[right])
            parents = unique_parents // 2


class PrioritizedReplayBuffer:
    """Replay buffer com amostragem proporcional ao erro TD (PER).

    Usa SumTree internamente para sample e update em O(log N).
    Suporta beta annealing: beta varia linearmente de beta_start a beta_end
    ao longo do treinamento.
    """

    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_end=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.buffer = []
        self.tree = SumTree(capacity)
        self.position = 0
        self._rng = np.random.default_rng()

    def set_total_steps(self, total_steps: int):
        """Define o número total de steps para o annealing do beta."""
        self._total_steps = total_steps

    def _get_beta(self, steps_done: int | None = None) -> float:
        """Calcula beta com annealing linear entre beta_start e beta_end."""
        if steps_done is None or not hasattr(self, '_total_steps') or self._total_steps is None or self._total_steps == 0:
            return self.beta_start
        fraction = min(1.0, steps_done / self._total_steps)
        return self.beta_start + (self.beta_end - self.beta_start) * fraction

    def push(self, state, action, reward, next_state, done):
        max_priority = self.tree.max_priority()
        # New transitions get the maximum priority (already alpha-exponentiated)
        priority = max_priority ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.tree.add(priority)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, steps_done: int | None = None):
        if len(self.buffer) == 0:
            raise IndexError("Cannot sample from an empty buffer")

        beta = self._get_beta(steps_done)
        indices = self.tree.sample(batch_size, self._rng)
        total = self.tree.total()
        total_n = self.tree.size

        # Probabilities P(i) = p_i / sum(p_j)
        probs = self.tree.sum_tree[self.tree.capacity + indices] / total

        # Importance-sampling correction: w_i = (N * P(i))^(-beta) / max_j(w_j)
        weights = (total_n * probs) ** (-beta)
        weights /= weights.max()

        batch = [self.buffer[idx] for idx in indices]
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
        # Vetoriza o cálculo do expoente alpha de uma só vez
        priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        # Dispara a atualização por camadas
        self.tree.batch_update(indices, priorities)

    def __len__(self):
        return self.tree.size


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
        # New transitions get the maximum priority (already alpha-exponentiated in update_priorities)
        priority = max_priority ** self.alpha

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = transition
            self.priorities[self.position] = priority

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[: len(self.buffer)])

        # Probabilities P(i) = p_i / sum(p_j), where p_i is already alpha-exponentiated in update_priorities/push
        probs = priorities / priorities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        total = len(self.buffer)
        # Importance-sampling correction: w_i = (N * P(i))^(-beta) / max_j(w_j)
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