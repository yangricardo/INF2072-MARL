"""Buffer de repetição com priorização (Prioritized Experience Replay).

Implementa SumTree (Segment Tree) para amostragem O(log N) em vez de O(N),
e beta annealing para correção de viés das importance-sampling weights.
"""

import numpy as np


class SumTree:
    """Árvore binária completa para soma de prioridades (Segment Tree / SumTree).

    Capacidade fixa alocada no construtor. Operações de sample e update em O(log N).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # tree[0] = raiz (soma total), tree[capacity-1] = último nó interno
        # tree[capacity..2*capacity-1] = folhas (prioridades individuais)
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write_index = 0
        self._size = 0

    def _propagate(self, idx: int):
        """Sobe pela árvore atualizando os pais após mudança em uma folha."""
        parent = idx // 2
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2

    def add(self, priority: float):
        """Adiciona/sobrescreve uma folha com a prioridade dada."""
        idx = self.capacity + self._write_index
        self.tree[idx] = priority
        self._propagate(idx)
        self._write_index = (self._write_index + 1) % self.capacity
        if self._size < self.capacity:
            self._size += 1

    def update(self, index: int, priority: float):
        """Atualiza a prioridade de uma folha pelo índice linear (buffer index)."""
        idx = self.capacity + index
        self.tree[idx] = priority
        self._propagate(idx)

    def sample(self, batch_size: int, rng: np.random.Generator) -> np.ndarray:
        """Amostra batch_size índices proporcionais às prioridades.

        Percorre a árvore da raiz até as folhas, usando um valor aleatório
        uniforme [0, total_sum) em cada percurso.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        total = self.total()
        if total == 0.0:
            return np.random.randint(0, self._size, size=batch_size)
        segment = total / batch_size
        for i in range(batch_size):
            # Amostra dentro do segmento i-ésimo para stratified sampling
            a = segment * i
            b = segment * (i + 1)
            value = rng.uniform(a, b)
            indices[i] = self._retrieve(value)
        return indices

    def _retrieve(self, value: float) -> int:
        """Desce na árvore a partir da raiz até encontrar a folha correspondente a 'value'."""
        idx = 1  # raiz
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - self.capacity  # converte de índice na árvore para índice do buffer

    def max_priority(self) -> float:
        """Retorna a maior prioridade armazenada (O(1) — lê a raiz do max-tree)."""
        # Percorre as folhas para encontrar o máximo — O(capacity) se usarmos isso.
        # Melhor: manter uma max-tree paralela, mas para simplificar, faremos scan
        # nas folhas. Como isso é chamado apenas no push (não no sample), o custo
        # é amortizado.
        if self._size == 0:
            return 1.0
        start = self.capacity
        end = start + self._size
        return float(self.tree[start:end].max())

    def total(self) -> float:
        """Soma de todas as prioridades (raiz da árvore)."""
        return float(self.tree[1])

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size


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
        probs = np.array(
            [self.tree.tree[self.tree.capacity + idx] / total for idx in indices],
            dtype=np.float64,
        )

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
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)

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