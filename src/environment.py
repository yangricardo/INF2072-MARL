"""Ambiente Warehouse multi-agente (porta da versão simples v1.3.0).

Extraído de: Código/Ambiente e Execução IDQN - Versão 1.3.0.py (classe WarehouseEnv).

Dois robôs numa grade devem pegar as caixas (``A``) e entregá-las nos alvos
(``B``). Esta variante **não** inclui falhas de atuação nem barreiras dinâmicas
(essas existem apenas nos experimentos em notebook). Mantém a captura de frame
de vídeo já corrigida (``buffer_rgba``).

Além da observação global compartilhada (``_get_observation``), oferece infra para
algoritmos de crítico centralizado: ``_get_global_state`` (estado global) e
``_get_observation_for_robot`` (observação local distinta por robô).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .config import MAP_CONFIG, IDQNConfig

# Número de ações por robô: 4 movimentos + pegar + soltar.
NUM_ACTIONS = 6


class WarehouseEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: str | None = None, seed: int | None = None, config=None):
        super().__init__()

        self.config = config or IDQNConfig()
        self.height = MAP_CONFIG["height"]
        self.width = MAP_CONFIG["width"]
        self.grid = [row[:] for row in MAP_CONFIG["grid"]]

        self.robot_positions: list[tuple[int, int]] = []
        self.box_positions: list[tuple[int, int] | None] = []
        self.targets = self._find_positions("B")

        self.num_robots = 2
        self.num_boxes = len(self._find_positions("A"))
        self.num_targets = len(self.targets)
        self.num_actions = NUM_ACTIONS

        self.delivered_boxes: list[bool] = []
        self.robot_carrying: dict[int, int | None] = {i: None for i in range(self.num_robots)}  # robot_id -> box_id (or None)
        self.steps = 0
        self.max_steps = self.config.MAX_STEPS
        self.total_deliveries = 0
        self.collisions = 0
        self.distance_traveled = [0, 0]
        self.previous_distances: list[int] = []

        self.action_space = spaces.Tuple(
            [spaces.Discrete(NUM_ACTIONS) for _ in range(self.num_robots)]
        )

        # robôs (x,y) + caixas (x,y) + alvos (x,y) + 2 features de distância por robô
        obs_dim = (
            (self.num_robots * 2)
            + (self.num_boxes * 2)
            + (self.num_targets * 2)
            + (self.num_robots * 2)
        )
        self.observation_space = spaces.Box(
            low=-1,
            high=self.height + self.width,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.seed_value = seed
        if seed is not None:
            self.seed(seed)

        self.fig = None
        self.ax = None
        self.frame_buffer = []

    def seed(self, seed=None):
        import random
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return [seed]

    def _find_positions(self, symbols) -> list[tuple[int, int]]:
        if isinstance(symbols, str):
            symbols = [symbols]
        positions = []
        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid[i][j]
                if any(cell.startswith(sym) for sym in symbols):
                    positions.append((i, j))
        return positions

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        self.grid = [row[:] for row in MAP_CONFIG["grid"]]
        self.robot_positions = self._find_positions("R")
        # list comprehension para o pyright inferir o tipo-união (vira None após pickup)
        self.box_positions = [pos for pos in self._find_positions("A")]
        self.delivered_boxes = [False] * self.num_boxes
        self.robot_carrying = {i: None for i in range(self.num_robots)}  # reset carrying status
        # Índices de caixas ativas (não entregues) — mantido por clareza,
        # mas tem impacto mínimo (apenas 4 caixas). A lista é atualizada
        # em _drop_box() quando uma caixa é entregue.
        self._active_box_indices = list(range(self.num_boxes))

        self.steps = 0
        self.total_deliveries = 0
        self.collisions = 0
        self.distance_traveled = [0, 0]
        self.frame_buffer = []

        self.previous_distances = [
            self._min_distance_to_boxes(r) for r in range(self.num_robots)
        ]

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _min_distance_to_boxes(self, robot_id):
        robot_pos = self.robot_positions[robot_id]
        min_dist = 100  
        for box_id in self._active_box_indices:
            box_pos = self.box_positions[box_id]
            if box_pos is not None and not self.delivered_boxes[box_id]:
                dist = abs(robot_pos[0] - box_pos[0]) + abs(robot_pos[1] - box_pos[1])
                if dist < min_dist:
                    min_dist = dist
        return 100 if min_dist == 100 else min_dist

    def _is_valid_position(self, pos, robot_id=None):
        i, j = pos
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return False

        cell = self.grid[i][j]
        if cell in ["X", "Y"]:
            return False

        if robot_id is not None:
            for rid, rpos in enumerate(self.robot_positions):
                if rid != robot_id and rpos == (i, j):
                    return False
        return True

    def _move_robot(self, robot_id, action):
        i, j = self.robot_positions[robot_id]
        new_i, new_j = i, j

        if action == 0:
            new_i -= 1
        elif action == 1:
            new_i += 1
        elif action == 2:
            new_j -= 1
        elif action == 3:
            new_j += 1

        if self._is_valid_position((new_i, new_j), robot_id):
            self.distance_traveled[robot_id] += 1
            old_pos = self.robot_positions[robot_id]
            self.grid[old_pos[0]][old_pos[1]] = "0"
            self.grid[new_i][new_j] = f"R{robot_id + 1}"
            self.robot_positions[robot_id] = (new_i, new_j)
            return -0.005
        else:
            self.collisions += 1
            return -0.05 # Punição severa para o robô odiar bater na parede

    def _pickup_box(self, robot_id):
        if self.robot_carrying[robot_id] is not None:
            # Robot already carrying a box; cannot pick up another
            return -0.02
        robot_pos = self.robot_positions[robot_id]
        for box_id, box_pos in enumerate(self.box_positions):
            if not self.delivered_boxes[box_id] and box_pos == robot_pos:
                self.box_positions[box_id] = None
                self.robot_carrying[robot_id] = box_id  # track which box this robot carries
                self.grid[robot_pos[0]][robot_pos[1]] = f"R{robot_id + 1}"
                return 2.0
        return -0.02

    def _drop_box(self, robot_id):
        robot_pos = self.robot_positions[robot_id]
        box_with_robot = self.robot_carrying[robot_id]

        if box_with_robot is None:
            return -0.02

        for target_pos in self.targets:
            if robot_pos == target_pos:
                self.delivered_boxes[box_with_robot] = True
                self.robot_carrying[robot_id] = None  # robot no longer carries this box
                self.total_deliveries += 1
                self.grid[robot_pos[0]][robot_pos[1]] = f"R{robot_id + 1}"
                return 25.0

        return -2.0

    def _calculate_shaped_reward(self, robot_id, base_reward):
        reward = base_reward

        # 1. Identifica o estado dinâmico: o que importa para este robô agora?
        box_with_robot = self.robot_carrying[robot_id]
        
        if box_with_robot is not None:
            # Se o robô ESTÁ carregando uma caixa, o objetivo dele é o alvo "B" mais próximo
            robot_pos = self.robot_positions[robot_id]
            current_distance = min(
                [
                    abs(robot_pos[0] - target_pos[0]) + abs(robot_pos[1] - target_pos[1])
                    for target_pos in self.targets
                ],
                default=100,
            )
        else:
            # Se o robô NÃO está carregando, o objetivo dele é a caixa "A" mais próxima do chão
            current_distance = self._min_distance_to_boxes(robot_id)

        # 2. Proteção de Transição: Se a ação do passo foi um Pick ou Drop de sucesso,
        # apenas atualizamos o histórico de distância sem aplicar diferenciais (evita distorções)
        if base_reward in [2.0, 25.0, -2.0]:
            self.previous_distances[robot_id] = current_distance
            return reward

        previous_distance = self.previous_distances[robot_id]

        # 3. Computa o ganho ou perda de proximidade em relação ao objetivo atual legítimo
        if current_distance < previous_distance:
            reward += 0.1 * (previous_distance - current_distance)
        elif current_distance > previous_distance:
            reward -= 0.02 * (current_distance - previous_distance)

        self.previous_distances[robot_id] = current_distance
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
            min_box_dist = min(
                [
                    abs(robot_pos[0] - box_pos[0]) + abs(robot_pos[1] - box_pos[1])
                    for box_id, box_pos in enumerate(self.box_positions)
                    if box_pos is not None
                    and not self.delivered_boxes[box_id]
                ],
                default=100,
            )
            obs.append(min_box_dist / (self.height + self.width))

            min_target_dist = min(
                [
                    abs(robot_pos[0] - target_pos[0]) + abs(robot_pos[1] - target_pos[1])
                    for target_pos in self.targets
                ],
                default=100,
            )
            obs.append(min_target_dist / (self.height + self.width))

        return np.array(obs, dtype=np.float32)

    def _get_global_state(self):
        """Estado global para críticos centralizados (QMIX/MAPPO/HATRPO).

        A observação completa já codifica o estado do armazém, então serve como
        estado global compartilhado.
        """
        return self._get_observation()

    def _get_observation_for_robot(self, robot_id):
        """Observação local distinta por robô: estado global + one-hot do id.

        Permite que algoritmos com obs por-agente diferenciem os robôs mesmo
        usando o mesmo estado global de base.

        Cache: _get_observation() é chamada uma vez por step e reutilizada para
        todos os robôs, evitando recálculo redundante O(n_robots * n_boxes).
        """
        # Reuse the base observation computed for this step (cache by step counter)
        if getattr(self, "_obs_cache_step", -1) != self.steps:
            self._obs_cache = self._get_observation()
            self._obs_cache_step = self.steps
        one_hot = np.zeros(self.num_robots, dtype=np.float32)
        one_hot[robot_id] = 1.0
        return np.concatenate([self._obs_cache, one_hot]).astype(np.float32)

    def step(self, actions):  # type: ignore[override]  # MARL: rewards é lista por agente
        self.steps += 1

        if len(actions) != self.num_robots:
            actions = [actions] * self.num_robots

        total_reward = 0.0
        rewards = [0.0, 0.0]

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

        # Identifica se houve uma mudança macro de estado no mapa para a equipa
        macro_change = any(r in [2.0, 25.0] for r in interaction_rewards)

        for robot_id in range(self.num_robots):
            base_reward = movement_rewards[robot_id] + interaction_rewards[robot_id]
            
            # CORREÇÃO CRÍTICA: Se o mapa mudou, recalibra o histórico de ambos para evitar picos
            if macro_change:
                box_with_robot = self.robot_carrying[robot_id]
                if box_with_robot is not None:
                    r_pos = self.robot_positions[robot_id]
                    current_dist = min([abs(r_pos[0]-t[0]) + abs(r_pos[1]-t[1]) for t in self.targets], default=100)
                else:
                    current_dist = self._min_distance_to_boxes(robot_id)
                self.previous_distances[robot_id] = current_dist

            shaped_reward = self._calculate_shaped_reward(robot_id, base_reward)
            rewards[robot_id] = shaped_reward
            total_reward += shaped_reward

        # Add terminal bonus once (not per robot) when all boxes delivered
        if all(self.delivered_boxes):
            total_reward += 50.0
            for robot_id in range(self.num_robots):
                rewards[robot_id] += 50.0 / self.num_robots  # distribute evenly

        terminated = all(self.delivered_boxes)
        truncated = self.steps >= self.max_steps

        observation = self._get_observation()
        info = self._get_info()

        return observation, rewards, terminated, truncated, info

    def _get_info(self):
        return {
            "steps": self.steps,
            "total_deliveries": self.total_deliveries,
            "collisions": self.collisions,
            "distance_traveled": self.distance_traveled.copy(),
            # métrica zerada (sem mecanismo de falhas) p/ compat. com código portado
            "failures": [0, 0],
            "remaining_boxes": sum(1 for d in self.delivered_boxes if not d),
            "success_rate": self.total_deliveries / self.num_boxes
            if self.steps > 0
            else 0,
            "all_delivered": all(self.delivered_boxes),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_array()
        return None

    def render_frame(self):
        """Renderiza um único frame como array RGB (método corrigido)."""
        fig, ax = plt.subplots(figsize=(10, 8))
        self._draw_grid(ax)

        fig.canvas.draw()

        try:
            # Método mais recente do matplotlib
            buffer = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
            img = np.asarray(buffer)
            img = img[:, :, :3]  # remove canal alpha
        except AttributeError:
            # Fallback para versões antigas
            fig.canvas.draw()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
            img = img.reshape(h, w, 4)[:, :, :3]

        plt.close(fig)
        return img

    def _draw_grid(self, ax):
        colors = {
            "0": "white",
            "X": "black",
            "Y": "gray",
            "R1": "red",
            "R2": "blue",
            "A": "orange",
            "B": "green",
        }

        for i in range(self.height):
            for j in range(self.width):
                cell = self.grid[i][j]
                color = colors.get(cell, "white")
                rect = Rectangle(
                    (j, self.height - 1 - i),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(rect)

                if cell in ["R1", "R2", "A", "B"]:
                    ax.text(
                        j + 0.5,
                        self.height - 0.5 - i,
                        cell,
                        ha="center",
                        va="center",
                        fontweight="bold",
                    )

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xticks(range(self.width))
        ax.set_yticks(range(self.height))
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Warehouse | Steps: {self.steps} | "
            f"Entregas: {self.total_deliveries}/{self.num_boxes}"
        )

    def _render_array(self):
        return self.render_frame()

    def get_frames(self):
        return self.frame_buffer.copy()

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
