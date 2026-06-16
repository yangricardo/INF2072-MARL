"""MAPPO — Multi-Agent PPO (atores por agente, crítico centralizado).

Extraído de: Código/Executa Experimento MPPO.ipynb
(Config, ActorNetwork, CriticNetwork, MAPPOAgent, train_mappo — 3ª versão).

On-policy: cada agente coleta uma trajetória por episódio e atualiza ator/crítico
com PPO (clip) + GAE. Adaptado ao env simples do ``src/``: o ator recebe a
observação por-robô (``_get_observation_for_robot``) e o crítico o estado global
(``_get_global_state``).
"""

import os
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from ..config import MAPPOConfig
from ..environment import WarehouseEnv
from ..evaluation import plot_consolidated_results, record_policy_video
from ..networks import ActorNetwork, CriticNetwork


class MAPPOAgent:
    def __init__(self, agent_id, state_dim, action_dim, config, global_state_dim):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.global_state_dim = global_state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(global_state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.global_states = []

        self.steps_done = 0
        self.epsilon = config.EPSILON_START

    def get_epsilon(self):
        return self.epsilon

    def decay_epsilon(self):
        self.epsilon = max(self.config.EPSILON_END, self.epsilon * self.config.EPSILON_DECAY)

    def select_action(self, state, global_state, training=True):
        self.steps_done += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs, _ = self.actor(state_tensor)
            # On-policy: sample from stochastic policy via categorical distribution
            dist = Categorical(probs)
            action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action, device=self.device)).item()
            return action, log_prob

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

    def compute_gae(self, rewards, values, dones):
        # Generalized Advantage Estimation (GAE-λ): Â_t = Σ_{l≥0}(γλ)^l·δ_{t+l}
        # δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
        # Uses append+reverse instead of insert(0) to avoid O(n²)
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.GAMMA * self.config.LAMBDA * (1 - dones[t]) * gae
            advantages.append(gae)
        advantages.reverse()
        return torch.stack(advantages).to(self.device)

    def update(self, global_states_all=None):
        """Atualiza ator/crítico via PPO. ``global_states_all`` é ignorado
        (mantido por compatibilidade com a chamada original)."""
        if len(self.states) == 0:
            return 0, 0

        n = len(self.states)
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        global_states = torch.FloatTensor(np.array(self.global_states)).to(self.device)

        with torch.no_grad():
            values = self.critic(global_states).squeeze(-1)
            advantages = self.compute_gae(rewards, values, dones)
            # Returns R_t = Â_t + V(s_t)
            returns = advantages + values

        # Advantage normalization (whitening): Â ← (Â - μ) / (σ + ε)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        for _ in range(self.config.PPO_EPOCHS):
            indices = np.random.permutation(n)
            for start in range(0, n, self.config.MINI_BATCH_SIZE):
                batch_indices = indices[start : start + self.config.MINI_BATCH_SIZE]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_global_states = global_states[batch_indices]

                probs, _ = self.actor(batch_states)
                new_log_probs = torch.log(
                    probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1) + 1e-10
                )

                # PPO probability ratio: r_t(θ) = π_θ(a|s) / π_{θ_old}(a|s) = exp(log π_new - log π_old)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                # PPO clip surrogate: L^CLIP = E[min(r·Â, clip(r,1-ε,1+ε)·Â)]
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS
                ) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Entropy bonus: H[π] = -Σ_a π(a|s)·log π(a|s)
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
                # Full objective: L = L^CLIP - c_H·H[π]
                actor_loss = actor_loss - self.config.ENTROPY_COEF * entropy

                batch_values = self.critic(batch_global_states).squeeze(-1)
                # Critic MSE loss: L_V = E[(V(s) - R_t)²]
                critic_loss = F.mse_loss(batch_values, batch_returns)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.MAX_GRAD_NORM
                )
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.MAX_GRAD_NORM
                )
                self.critic_optimizer.step()

                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()

        self.decay_epsilon()
        self.clear_memory()
        denom = max(1, n // self.config.MINI_BATCH_SIZE)
        return total_actor_loss / denom, total_critic_loss / denom


def run(config=None, num_sessions=1, record_video=True):
    """Treina o MAPPO no WarehouseEnv simples e salva métricas/plot/vídeo."""
    config = config or MAPPOConfig()
    total_episodes = config.EPISODES_PER_SESSION * num_sessions

    print("=" * 80)
    print("🤖 TREINAMENTO MAPPO — Multi-Agent PPO (crítico centralizado)")
    print("=" * 80)
    print(f"   • Episódios: {total_episodes} | Max steps/ep: {config.MAX_STEPS}")

    base_dir = Path(config.BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    env = WarehouseEnv(config=config)
    env.reset()
    n_agents = env.num_robots
    state_dim = len(env._get_observation_for_robot(0))
    global_state_dim = len(env._get_global_state())
    action_dim = env.num_actions

    agents = [
        MAPPOAgent(i, state_dim, action_dim, config, global_state_dim)
        for i in range(n_agents)
    ]

    metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
    }

    # Persistent ThreadPoolExecutor for select_action and update — NOT created per step
    _mappo_pool = ThreadPoolExecutor(max_workers=n_agents, thread_name_prefix="mappo")

    for ep in tqdm(range(total_episodes), desc="MAPPO"):
        env.reset()
        for agent in agents:
            agent.clear_memory()
        global_state = env._get_global_state()
        ep_reward = 0.0
        step = 0
        info = env._get_info()
        for step in range(config.MAX_STEPS):
            local_obs = [env._get_observation_for_robot(i) for i in range(n_agents)]

            # select_action in parallel — each agent has independent network and state
            action_futures = [
                _mappo_pool.submit(agents[i].select_action, local_obs[i], global_state, True)
                for i in range(n_agents)
            ]
            results = [f.result() for f in action_futures]
            actions = [r[0] for r in results]
            log_probs = [r[1] for r in results]

            _, rewards, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            for i, agent in enumerate(agents):
                agent.store_transition(
                    local_obs[i], actions[i], log_probs[i], rewards[i], done, global_state
                )
            ep_reward += sum(rewards)
            global_state = env._get_global_state()
            if done:
                break

        # PPO update in parallel — each agent has independent actor + critic
        update_futures = [_mappo_pool.submit(agent.update) for agent in agents]
        wait(update_futures)

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
                f"ε: {agents[0].get_epsilon():.3f}"
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

    def _act(env, _obs):
        gs = env._get_global_state()
        return [
            agents[i].select_action(
                env._get_observation_for_robot(i), gs, training=False
            )[0]
            for i in range(n_agents)
        ]

    video_path = None
    if record_video:
        results_dir = base_dir / "final_results"
        os.makedirs(results_dir, exist_ok=True)
        video_path = record_policy_video(config, _act, results_dir)

    return agents, metrics, video_path
