"""QMIX — mixing network monotônico sobre o estado global.

Extraído de: Código/Executa - Experimento.ipynb
(Config, QMIXPrioritizedReplayBuffer, DQN, QMixer, QMIXAgent, QMIXTrainer,
train_single_session). A rede por agente (``DQN``) é idêntica ao ``ImprovedDQN``,
por isso é reutilizada. O estado global vem de ``env._get_global_state()``.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm

from ..config import QMIXConfig
from ..environment import WarehouseEnv
from ..evaluation import plot_consolidated_results, record_policy_video
from ..networks import ImprovedDQN, QMixer
from ..replay_buffer import QMIXPrioritizedReplayBuffer


class QMIXAgent:
    """Agente Q individual (rede ``ImprovedDQN``) usado dentro do mixer QMIX."""

    def __init__(self, agent_id, state_dim, action_dim, config, n_agents, global_state_dim):
        self.agent_id = agent_id
        self.action_dim = action_dim
        self.config = config
        self.n_agents = n_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        self.steps_done = 0
        self.learning_steps = 0
        self.total_episodes = 0
        self.losses = []

    def get_epsilon(self):
        if self.steps_done >= self.config.EPSILON_DECAY_STEPS:
            return self.config.EPSILON_END
        epsilon = self.config.EPSILON_START - (
            self.config.EPSILON_START - self.config.EPSILON_END
        ) * self.steps_done / self.config.EPSILON_DECAY_STEPS
        return max(self.config.EPSILON_END, epsilon)

    def select_action(self, state, training=True):
        self.steps_done += 1
        if training and np.random.random() < self.get_epsilon():
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return int(q_values.argmax().item())

    def get_q_values(self, states):
        if isinstance(states, np.ndarray):
            states_tensor = torch.FloatTensor(states).to(self.device)
        else:
            states_tensor = states.to(self.device)
        return self.policy_net(states_tensor)

    def get_target_q_values(self, states):
        if isinstance(states, np.ndarray):
            states_tensor = torch.FloatTensor(states).to(self.device)
        else:
            states_tensor = states.to(self.device)
        return self.target_net(states_tensor)

    def soft_update_target(self):
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.config.TAU * policy_param.data
                + (1 - self.config.TAU) * target_param.data
            )


class QMIXTrainer:
    """Treina as redes individuais + o mixer com o alvo conjunto QMIX."""

    def __init__(self, agents, mixer, config, state_dim, global_state_dim):
        self.agents = agents
        self.mixer = mixer
        self.config = config
        self.state_dim = state_dim
        self.global_state_dim = global_state_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = QMIXPrioritizedReplayBuffer(config.BUFFER_SIZE, alpha=config.ALPHA)
        self.mixer_optimizer = optim.Adam(mixer.parameters(), lr=config.LEARNING_RATE)
        self.learning_steps = 0

    def remember(self, state, actions, rewards, next_state, done, global_state, next_global_state):
        self.memory.push(
            state, actions, rewards, next_state, done, global_state, next_global_state
        )

    def optimize(self):
        if len(self.memory) < self.config.BATCH_SIZE:
            return 0

        self.learning_steps += 1
        if self.learning_steps % self.config.TRAIN_FREQ != 0:
            return 0

        (states, actions, rewards, next_states, dones,
         global_states, next_global_states, indices, weights) = self.memory.sample(
            self.config.BATCH_SIZE
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        global_states = torch.FloatTensor(global_states).to(self.device)
        next_global_states = torch.FloatTensor(next_global_states).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        curr_qs = []
        curr_qs_all = []  # cache all Q-values for reuse in per-agent loss
        for i, agent in enumerate(self.agents):
            agent_qs = agent.get_q_values(states)
            curr_qs_all.append(agent_qs)
            curr_qs.append(agent_qs.gather(1, actions[:, i].unsqueeze(1)))
        curr_qs = torch.cat(curr_qs, dim=1)

        target_qs = []
        for agent in self.agents:
            agent_target_qs = agent.get_target_q_values(next_states)
            target_qs.append(agent_target_qs.max(1, keepdim=True)[0])
        target_qs = torch.cat(target_qs, dim=1)

        curr_q_total = self.mixer(curr_qs, global_states)

        with torch.no_grad():
            target_q_total = self.mixer(target_qs, next_global_states)
            target = rewards.sum(dim=1, keepdim=True) + self.config.GAMMA * target_q_total * (
                1 - dones.unsqueeze(1)
            )

        td_errors = (target - curr_q_total).squeeze()
        loss = (weights * td_errors.pow(2)).mean()

        self.mixer_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.config.MAX_GRAD_NORM)
        self.mixer_optimizer.step()

        for i, agent in enumerate(self.agents):
            agent.optimizer.zero_grad()
            # Reuse cached Q-values instead of calling get_q_values again
            agent_curr_qs = curr_qs_all[i].gather(1, actions[:, agent.agent_id].unsqueeze(1))
            agent_loss = (weights * (target - curr_q_total).detach() * agent_curr_qs).mean()
            agent_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.policy_net.parameters(), self.config.MAX_GRAD_NORM
            )
            agent.optimizer.step()
            agent.soft_update_target()

        priorities = td_errors.abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities)

        return loss.item()


def run(config=None, num_sessions=1, record_video=True):
    """Treina o QMIX no WarehouseEnv simples e salva métricas/plot/vídeo."""
    config = config or QMIXConfig()
    total_episodes = config.EPISODES_PER_SESSION * num_sessions

    print("=" * 80)
    print("🤖 TREINAMENTO QMIX — mixing network monotônico")
    print("=" * 80)
    print(f"   • Episódios: {total_episodes} | Max steps/ep: {config.MAX_STEPS}")

    base_dir = Path(config.BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = WarehouseEnv(config=config)
    obs, _ = env.reset()
    state_dim = len(obs)
    global_state_dim = len(env._get_global_state())
    action_dim = env.num_actions
    n_agents = env.num_robots

    agents = [
        QMIXAgent(i, state_dim, action_dim, config, n_agents, global_state_dim)
        for i in range(n_agents)
    ]
    mixer = QMixer(n_agents, global_state_dim, config.MIXER_HIDDEN_DIM).to(device)
    trainer = QMIXTrainer(agents, mixer, config, state_dim, global_state_dim)

    metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
    }

    # Persistent ThreadPoolExecutor for select_action — NOT created per step
    _qmix_pool = ThreadPoolExecutor(max_workers=len(agents), thread_name_prefix="qmix")

    for ep in tqdm(range(total_episodes), desc="QMIX"):
        obs, info = env.reset()
        global_state = env._get_global_state()
        ep_reward = 0.0
        step = 0
        for step in range(config.MAX_STEPS):
            # select_action in parallel — each agent has independent network and state
            action_futures = [_qmix_pool.submit(a.select_action, obs) for a in agents]
            actions = [f.result() for f in action_futures]
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_global_state = env._get_global_state()
            trainer.remember(
                obs, actions, rewards, next_obs, terminated or truncated,
                global_state, next_global_state,
            )
            trainer.optimize()
            ep_reward += sum(rewards)
            obs = next_obs
            global_state = next_global_state
            if terminated or truncated:
                break

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

    video_path = None
    if record_video:
        results_dir = base_dir / "final_results"
        os.makedirs(results_dir, exist_ok=True)
        video_path = record_policy_video(
            config,
            lambda env, obs: [a.select_action(obs, training=False) for a in agents],
            results_dir,
        )

    return trainer, metrics, video_path
