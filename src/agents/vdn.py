"""VDN — Value Decomposition Networks (Q_total = Q1 + Q2).

Extraído de: Código/Executa Experimento VDN.ipynb (VDNController, run_vdn_training).

Aprendizado centralizado / execução descentralizada: um único otimizador treina
as duas redes Q individuais com perda conjunta sobre a observação global
compartilhada do ``WarehouseEnv``. Adaptado para o ambiente simples do ``src/``
(sem falhas) e para as utilidades de saída comuns (CSV/plot/vídeo).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from ..config import VDNConfig
from ..environment import WarehouseEnv
from ..evaluation import plot_consolidated_results, record_policy_video
from ..networks import AgentNet
from ..replay_buffer import VDNPrioritizedReplayBuffer


class VDNController:
    """Controlador VDN centralizado: ``Q_total = Q_1 + Q_2``."""

    def __init__(self, n_agents, state_dim, action_dim, config):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_nets = nn.ModuleList(
            [
                AgentNet(state_dim, action_dim, config.HIDDEN_DIM, config.DROPOUT_RATE)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        self.target_nets = nn.ModuleList(
            [
                AgentNet(state_dim, action_dim, config.HIDDEN_DIM, config.DROPOUT_RATE)
                for _ in range(n_agents)
            ]
        ).to(self.device)
        for i in range(n_agents):
            self.target_nets[i].load_state_dict(self.policy_nets[i].state_dict())
            self.target_nets[i].eval()

        self.optimizer = optim.Adam(
            self.policy_nets.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPISODES_TOTAL, eta_min=1e-5
        )

        self.memory = VDNPrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.ALPHA)
        self.steps_done = 0
        self.learning_steps = 0
        self.losses = []

    def _get_beta(self):
        frac = min(
            1.0, self.steps_done / (self.config.EPISODES_TOTAL * self.config.MAX_STEPS)
        )
        return self.config.BETA_START + frac * (
            self.config.BETA_END - self.config.BETA_START
        )

    def get_epsilon(self):
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END
        t = self.steps_done / self.config.EPSILON_DECAY_STEPS
        return self.config.EPSILON_START + t * (
            self.config.EPSILON_END - self.config.EPSILON_START
        )

    def select_actions(self, obs, training=True):
        """obs: observação global compartilhada; devolve uma ação por agente."""
        self.steps_done += 1
        eps = self.get_epsilon() if training else 0.0
        actions = []
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            for net in self.policy_nets:
                if training and np.random.random() < eps:
                    actions.append(np.random.randint(self.action_dim))
                else:
                    net.eval()
                    q = net(obs_t)
                    if training:
                        net.train()
                    actions.append(int(q.argmax(dim=1).item()))
        return actions

    def remember(self, state, actions, rewards, next_state, done):
        self.memory.push(state, actions, rewards, next_state, done)

    def optimize(self):
        if (
            len(self.memory) < self.config.BATCH_SIZE
            or self.steps_done < self.config.LEARNING_STARTS
        ):
            return 0.0

        self.learning_steps += 1
        if self.learning_steps % self.config.TRAIN_FREQ != 0:
            return 0.0

        beta = self._get_beta()
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.config.BATCH_SIZE, beta=beta)
        )

        S = torch.FloatTensor(states).to(self.device)
        A = torch.LongTensor(actions).to(self.device)
        R = torch.FloatTensor(rewards).to(self.device)
        S_ = torch.FloatTensor(next_states).to(self.device)
        D = torch.FloatTensor(dones).to(self.device)
        W = torch.FloatTensor(weights).to(self.device)

        # Q_total atual
        q_total = torch.zeros(self.config.BATCH_SIZE, device=self.device)
        for i, net in enumerate(self.policy_nets):
            q_vals = net(S)
            q_a = q_vals.gather(1, A[:, i].unsqueeze(1)).squeeze(1)
            q_total = q_total + q_a

        # Q_total target (Double DQN por agente)
        with torch.no_grad():
            q_total_next = torch.zeros(self.config.BATCH_SIZE, device=self.device)
            for pnet, tnet in zip(self.policy_nets, self.target_nets):
                next_a = pnet(S_).argmax(dim=1, keepdim=True)
                next_q = tnet(S_).gather(1, next_a).squeeze(1)
                q_total_next = q_total_next + next_q
            r_total = R.sum(dim=1)
            y = r_total + self.config.GAMMA * q_total_next * (1 - D)

        td_errors = (y - q_total).detach().cpu().numpy()
        loss = (W * (y - q_total).pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.policy_nets.parameters(), self.config.MAX_GRAD_NORM
        )
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

        if self.config.USE_SOFT_UPDATE:
            self._soft_update()
        elif self.learning_steps % self.config.TARGET_UPDATE_FREQ == 0:
            self._hard_update()

        loss_val = loss.item()
        self.losses.append(loss_val)
        return loss_val

    def _soft_update(self):
        # Polyak averaging (soft update) — vetorizado via _foreach_lerp_
        # para cada par (policy_net, target_net)
        for pnet, tnet in zip(self.policy_nets, self.target_nets):
            torch._foreach_lerp_(
                list(tnet.parameters()),
                list(pnet.parameters()),
                self.config.TAU,
            )

    def _hard_update(self):
        for pnet, tnet in zip(self.policy_nets, self.target_nets):
            tnet.load_state_dict(pnet.state_dict())


def run(config=None, num_sessions=1, record_video=True):
    """Treina o VDN no WarehouseEnv simples e salva métricas/plot/vídeo."""
    config = config or VDNConfig()
    total_episodes = config.EPISODES_PER_SESSION * num_sessions
    config.EPISODES_TOTAL = total_episodes

    print("=" * 80)
    print("🤖 TREINAMENTO VDN — Value Decomposition Networks (Q_total = Q1 + Q2)")
    print("=" * 80)
    print(f"   • Episódios: {total_episodes} | Max steps/ep: {config.MAX_STEPS}")

    base_dir = Path(config.BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    env = WarehouseEnv(config=config)
    obs, _ = env.reset()
    state_dim = len(obs)
    action_dim = env.num_actions
    controller = VDNController(2, state_dim, action_dim, config)

    metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
    }

    for ep in tqdm(range(total_episodes), desc="VDN"):
        obs, info = env.reset()
        ep_reward = 0.0
        step = 0
        for step in range(config.MAX_STEPS):
            actions = controller.select_actions(obs, training=True)
            nobs, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            controller.remember(obs, actions, rewards, nobs, float(done))
            controller.optimize()
            ep_reward += sum(rewards)
            obs = nobs
            if done:
                break
        controller.scheduler.step()

        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_deliveries"].append(info["total_deliveries"])
        metrics["episode_steps"].append(step + 1)
        metrics["success_rates"].append(info["success_rate"])
        metrics["collisions"].append(info["collisions"])

        if (ep + 1) % 100 == 0:
            rr = metrics["episode_rewards"][-100:]
            rd = metrics["episode_deliveries"][-100:]
            print(
                f"Ep {ep + 1:4d}/{total_episodes} | Reward: {np.mean(rr):8.2f} | "
                f"Entregas: {np.mean(rd):.2f}/{env.num_boxes} | "
                f"ε: {controller.get_epsilon():.3f}"
            )

    env.close()

    consolidated_dir = base_dir / "consolidated_results"
    os.makedirs(consolidated_dir, exist_ok=True)
    pd.DataFrame(
        {
            "episode": range(1, len(metrics["episode_rewards"]) + 1),
            "reward": metrics["episode_rewards"],
            "deliveries": metrics["episode_deliveries"],
            "steps": metrics["episode_steps"],
            "success_rate": metrics["success_rates"],
            "collisions": metrics["collisions"],
        }
    ).to_csv(consolidated_dir / "consolidated_metrics.csv", index=False)
    plot_consolidated_results(metrics, consolidated_dir)
    print(f"\n📁 Resultados consolidados salvos em: {consolidated_dir}")

    video_path = None
    if record_video:
        results_dir = base_dir / "final_results"
        os.makedirs(results_dir, exist_ok=True)
        video_path = record_policy_video(
            config,
            lambda env, obs: controller.select_actions(obs, training=False),
            results_dir,
        )

    return controller, metrics, video_path
