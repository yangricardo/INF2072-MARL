"""HATRPO — atores por agente (PPO clip) + crítico centralizado.

Extraído de: Código/Executa Experimento HATRPO.ipynb
(HATRPOConfigOptimized, ImprovedActorNetwork, ImprovedCriticNetwork,
HATRPOAgentOptimized, CentralizedCriticOptimized, TrajectoryBuffer,
run_hatrpo_training).

On-policy: por episódio coleta uma trajetória, computa GAE com o crítico
centralizado (estado global) e atualiza cada ator com a vantagem compartilhada.
Adaptado ao env simples do ``src/``: estados por-robô via
``_get_observation_for_robot`` e estado global via ``_get_global_state``.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..config import HATRPOConfig
from ..environment import WarehouseEnv
from ..evaluation import plot_consolidated_results, record_policy_video
from ..networks import ImprovedActorNetwork, ImprovedCriticNetwork


class HATRPOAgentOptimized:
    def __init__(self, agent_id, state_dim, action_dim, global_state_dim, config):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = ImprovedActorNetwork(
            state_dim, action_dim, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE
        ).to(self.device)
        self.actor_old = ImprovedActorNetwork(
            state_dim, action_dim, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE
        ).to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)

        self.steps_done = 0
        self.epsilon = config.EPSILON_START
        self.total_episodes = 0

    def get_epsilon(self):
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END
        epsilon = self.config.EPSILON_START - (
            self.config.EPSILON_START - self.config.EPSILON_END
        ) * self.steps_done / self.config.EPSILON_DECAY_STEPS
        return max(self.config.EPSILON_END, epsilon)

    def select_action(self, state, training=True):
        self.steps_done += 1
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            probs, _ = self.actor(state_tensor)
            if training and np.random.random() < self.get_epsilon():
                action = np.random.randint(self.action_dim)
            else:
                action = int(probs.argmax().item())
            log_prob = torch.log(probs[0, action] + 1e-10).item()
            return action, log_prob

    def update_actor(self, states, actions, advantages, old_log_probs):
        """Atualiza o ator com PPO clip + bônus de entropia."""
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)

        probs, _ = self.actor(states_tensor)
        new_log_probs = torch.log(
            probs.gather(1, actions_tensor.unsqueeze(1)).squeeze(1) + 1e-10
        )

        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        surr1 = ratio * advantages_tensor
        clip_eps = self.config.CLIP_EPS
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()
        total_loss = policy_loss - self.config.ENTROPY_COEF * entropy

        self.actor_optimizer.zero_grad()
        total_loss.backward()
        if self.config.USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.MAX_GRAD_NORM)
        self.actor_optimizer.step()

        if self.steps_done % self.config.TARGET_UPDATE_FREQ == 0:
            self.actor_old.load_state_dict(self.actor.state_dict())

        return total_loss.item()


class CentralizedCriticOptimized:
    def __init__(self, global_state_dim, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.critic = ImprovedCriticNetwork(
            global_state_dim, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE
        ).to(self.device)
        self.critic_target = ImprovedCriticNetwork(
            global_state_dim, config.HIDDEN_DIM, config.NUM_LAYERS, config.DROPOUT_RATE
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

    def get_value(self, global_state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
            return self.critic(state_tensor).item()

    def get_values(self, global_states):
        with torch.no_grad():
            states_tensor = torch.FloatTensor(global_states).to(self.device)
            return self.critic(states_tensor).squeeze(-1).cpu().numpy()

    def update(self, global_states, returns):
        states_tensor = torch.FloatTensor(global_states).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        values = self.critic(states_tensor).squeeze(-1)
        critic_loss = F.mse_loss(values, returns_tensor)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.USE_GRADIENT_CLIPPING:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.MAX_GRAD_NORM)
        self.critic_optimizer.step()

        tau = self.config.TAU
        for target_param, param in zip(
            self.critic_target.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_((1 - tau) * target_param.data + tau * param.data)

        return critic_loss.item()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.GAMMA * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.config.GAMMA * self.config.LAMBDA * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = np.array(advantages, dtype=np.float32)
        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns


class TrajectoryBuffer:
    def __init__(self):
        self.clear()

    def push(self, state, action, reward, next_state, done, log_prob, value, global_state, next_global_state):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.global_states.append(global_state)
        self.next_global_states.append(next_global_state)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.global_states = []
        self.next_global_states = []

    def get_trajectory(self):
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "dones": np.array(self.dones),
            "log_probs": np.array(self.log_probs),
            "values": np.array(self.values),
            "global_states": np.array(self.global_states),
        }

    def __len__(self):
        return len(self.states)


def run(config=None, num_sessions=1, record_video=True):
    """Treina o HATRPO no WarehouseEnv simples e salva métricas/plot/vídeo."""
    config = config or HATRPOConfig()
    total_episodes = config.EPISODES_PER_SESSION * num_sessions

    print("=" * 80)
    print("🤖 TREINAMENTO HATRPO — trust-region + crítico centralizado")
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
        HATRPOAgentOptimized(i, state_dim, action_dim, global_state_dim, config)
        for i in range(n_agents)
    ]
    critic = CentralizedCriticOptimized(global_state_dim, config)

    metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
    }

    for ep in tqdm(range(total_episodes), desc="HATRPO"):
        env.reset()
        all_states = np.stack([env._get_observation_for_robot(i) for i in range(n_agents)])
        global_state = env._get_global_state()
        ep_reward = 0.0
        step = 0
        info = env._get_info()
        buf = TrajectoryBuffer()

        for step in range(config.MAX_STEPS):
            actions, log_probs = [], []
            for i, agent in enumerate(agents):
                action, log_prob = agent.select_action(all_states[i], training=True)
                actions.append(action)
                log_probs.append(log_prob)

            value = critic.get_value(global_state)
            _, rewards, terminated, truncated, info = env.step(actions)
            next_all_states = np.stack(
                [env._get_observation_for_robot(i) for i in range(n_agents)]
            )
            next_global_state = env._get_global_state()

            buf.push(
                all_states.flatten(), np.array(actions), sum(rewards),
                next_all_states.flatten(), terminated or truncated,
                np.array(log_probs), value, global_state, next_global_state,
            )
            ep_reward += sum(rewards)
            all_states = next_all_states
            global_state = next_global_state
            if terminated or truncated:
                break

        traj = buf.get_trajectory()
        if len(traj["states"]) > 0:
            values = critic.get_values(traj["global_states"])
            advantages, returns = critic.compute_gae(
                traj["rewards"], values, traj["dones"]
            )
            if len(advantages) > 1 and advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            critic.update(traj["global_states"], returns)
            for i, agent in enumerate(agents):
                agent_states = traj["states"][:, i * state_dim : (i + 1) * state_dim]
                agent_actions = traj["actions"][:, i]
                agent_old_log_probs = traj["log_probs"][:, i]
                agent.update_actor(
                    agent_states, agent_actions, advantages, agent_old_log_probs
                )

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
        return [
            agents[i].select_action(env._get_observation_for_robot(i), training=False)[0]
            for i in range(n_agents)
        ]

    video_path = None
    if record_video:
        results_dir = base_dir / "final_results"
        os.makedirs(results_dir, exist_ok=True)
        video_path = record_policy_video(config, _act, results_dir)

    return agents, metrics, video_path
