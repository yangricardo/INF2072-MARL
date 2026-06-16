"""MAPPO — Multi-Agent PPO canônico (Yu et al. 2022).

CTDE com **parameter sharing**: um único **ator compartilhado** `π_θ(a|o_i)` para
todos os robôs homogêneos (a obs por-robô traz one-hot de id + flag de inventário
para quebrar simetria) e um único **crítico centralizado** `V_φ(s)` sobre o estado
global, treinado no **team return** (`r_team = Σ_i r_i`).

Reescrito a partir da versão por-agente anterior (que tinha crítico por-agente,
GAE tratando *truncation* como *termination*, avaliação estocástica e referência a
`config.EPSILON_START` — campo já removido do `MAPPOConfig`, o que quebrava a
instanciação). Detalhes em REVIEW.md (Fase 11, MP0–MP6).

Estrutura espelha `VDNController`: um controlador central + um `run()` com o loop de
episódios, métricas, CSV, plot e vídeo.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm

from ..config import DEVICE, MAPPOConfig
from ..environment import WarehouseEnv
from ..evaluation import plot_consolidated_results, record_policy_video
from ..networks import ActorNetwork, CriticNetwork


class MAPPOController:
    """MAPPO CTDE: ator compartilhado (parameter sharing) + crítico centralizado único.

    Coleta ``ROLLOUT_EPISODES`` episódios por atualização (reduz a variância do
    update on-policy) e roda PPO-clip com GAE sobre o ``team_reward``, value-clipping
    e early-stop opcional por KL aproximado.
    """

    def __init__(self, n_agents, state_dim, action_dim, global_state_dim, config):
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.config = config
        self.device = DEVICE

        # Parameter sharing: UM ator para todos os robôs homogêneos.
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        # Crítico centralizado ÚNICO sobre o estado global.
        self.critic = CriticNetwork(global_state_dim).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # Rollout: lista de episódios coletados desde o último update.
        self._episodes: list[dict] = []
        self._cur: dict = {}
        self.steps_done = 0

    # ------------------------------------------------------------------ coleta
    def start_episode(self):
        self._cur = {
            "obs": [],            # por passo: [n_agents, state_dim]
            "actions": [],        # [n_agents]
            "log_probs": [],      # [n_agents]
            "global_state": [],   # [global_state_dim]
            "value": [],          # V(s_t) escalar (na coleta)
            "team_reward": [],    # escalar Σ_i r_i
        }

    @torch.no_grad()
    def select_actions(self, local_obs, global_state, training=True):
        """``local_obs``: lista de ``n_agents`` observações por-robô.

        Treino: amostra da política (estocástico) e registra obs/ação/logprob/valor/
        estado global do passo. Avaliação: devolve ``argmax`` determinístico e não
        muta estado nem ``steps_done``.
        """
        obs_t = torch.as_tensor(np.asarray(local_obs), dtype=torch.float32, device=self.device)
        probs, _ = self.actor(obs_t)  # [n_agents, action_dim]

        if not training:
            return [int(a) for a in probs.argmax(dim=1).tolist()]

        dist = Categorical(probs)
        actions = dist.sample()                  # [n_agents]
        log_probs = dist.log_prob(actions)       # [n_agents]

        gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        value = self.critic(gs_t).squeeze(-1).item()  # crítico centralizado V(s_t)

        self.steps_done += self.n_agents
        self._cur["obs"].append(np.asarray(local_obs, dtype=np.float32))
        self._cur["actions"].append(actions.cpu().numpy())
        self._cur["log_probs"].append(log_probs.cpu().numpy())
        self._cur["global_state"].append(np.asarray(global_state, dtype=np.float32))
        self._cur["value"].append(value)
        return [int(a) for a in actions.tolist()]

    def record_reward(self, team_reward):
        """Registra o team reward (Σ_i r_i) do passo recém-executado."""
        self._cur["team_reward"].append(float(team_reward))

    def finish_episode(self, last_global_state, terminated):
        """Fecha o episódio. ``terminated`` = terminal real (todas entregues);
        em *truncation* (timeout) bootstrapa com ``V(last_global_state)``."""
        if terminated:
            last_value = 0.0
        else:
            with torch.no_grad():
                gs_t = torch.as_tensor(
                    last_global_state, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                last_value = self.critic(gs_t).squeeze(-1).item()
        self._cur["last_value"] = last_value
        self._cur["terminated"] = bool(terminated)
        self._episodes.append(self._cur)
        self._cur = {}

    # ----------------------------------------------------------------- update
    def _compute_gae(self, rewards, values, last_value, terminated):
        # GAE-λ: Â_t = Σ_l (γλ)^l δ_{t+l}, δ_t = r_t + γ·V(s_{t+1})·(1-term) - V(s_t).
        # Bootstrap correto: usa ``terminated`` (não ``truncated``) e V(s_T) no timeout.
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        next_nonterminal = 0.0 if terminated else 1.0
        for t in reversed(range(T)):
            delta = rewards[t] + self.config.GAMMA * next_value * next_nonterminal - values[t]
            gae = delta + self.config.GAMMA * self.config.LAMBDA * next_nonterminal * gae
            adv[t] = gae
            next_value = values[t]
            next_nonterminal = 1.0  # passos interiores nunca são terminais
        returns = adv + np.asarray(values, dtype=np.float32)
        return adv, returns

    def update(self):
        """Atualiza ator/crítico via PPO sobre os episódios coletados no rollout."""
        if not self._episodes:
            return 0.0, 0.0

        # Pools: ator por (passo × agente); crítico por passo (estado global).
        a_obs, a_act, a_logp, a_adv = [], [], [], []
        c_gs, c_ret, c_vold = [], [], []
        for ep in self._episodes:
            T = len(ep["team_reward"])
            if T == 0:
                continue
            values = np.asarray(ep["value"][:T], dtype=np.float32)
            adv, returns = self._compute_gae(ep["team_reward"], values, ep["last_value"], ep["terminated"])

            obs = np.asarray(ep["obs"][:T], dtype=np.float32)      # [T, n, sdim]
            acts = np.asarray(ep["actions"][:T])                   # [T, n]
            logp = np.asarray(ep["log_probs"][:T], dtype=np.float32)  # [T, n]

            a_obs.append(obs.reshape(T * self.n_agents, -1))
            a_act.append(acts.reshape(T * self.n_agents))
            a_logp.append(logp.reshape(T * self.n_agents))
            a_adv.append(np.repeat(adv, self.n_agents))            # vantagem compartilhada
            c_gs.append(np.asarray(ep["global_state"][:T], dtype=np.float32))
            c_ret.append(returns)
            c_vold.append(values)

        if not a_obs:
            self._episodes = []
            return 0.0, 0.0

        a_obs = torch.as_tensor(np.concatenate(a_obs), dtype=torch.float32, device=self.device)
        a_act = torch.as_tensor(np.concatenate(a_act), dtype=torch.long, device=self.device)
        a_logp = torch.as_tensor(np.concatenate(a_logp), dtype=torch.float32, device=self.device)
        a_adv = torch.as_tensor(np.concatenate(a_adv), dtype=torch.float32, device=self.device)
        # Whitening das vantagens: Â ← (Â - μ) / (σ + ε)
        a_adv = (a_adv - a_adv.mean()) / (a_adv.std() + 1e-8)

        c_gs = torch.as_tensor(np.concatenate(c_gs), dtype=torch.float32, device=self.device)
        c_ret = torch.as_tensor(np.concatenate(c_ret), dtype=torch.float32, device=self.device)
        c_vold = torch.as_tensor(np.concatenate(c_vold), dtype=torch.float32, device=self.device)

        n_as = a_obs.shape[0]
        n_t = c_gs.shape[0]
        mb = self.config.MINI_BATCH_SIZE
        clip = self.config.CLIP_EPS
        target_kl = getattr(self.config, "TARGET_KL", None)

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        n_updates = 0
        for _ in range(self.config.PPO_EPOCHS):
            # ---- ator (amostras por passo × agente) ----
            perm = torch.randperm(n_as, device=self.device)
            kl_sum, kl_n = 0.0, 0
            for s in range(0, n_as, mb):
                idx = perm[s : s + mb]
                probs, _ = self.actor(a_obs[idx])
                dist = Categorical(probs)
                new_logp = dist.log_prob(a_act[idx])
                logratio = new_logp - a_logp[idx]
                ratio = torch.exp(logratio)
                adv = a_adv[idx]
                # PPO clip: L^CLIP = E[min(r·Â, clip(r,1-ε,1+ε)·Â)]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv
                entropy = dist.entropy().mean()
                actor_loss = -torch.min(surr1, surr2).mean() - self.config.ENTROPY_COEF * entropy

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.MAX_GRAD_NORM)
                self.actor_optimizer.step()

                with torch.no_grad():
                    # KL aproximado (Schulman): E[(r-1) - log r] ≥ 0
                    kl_sum += ((ratio - 1) - logratio).mean().item()
                    kl_n += 1
                total_actor_loss += actor_loss.item()
                n_updates += 1

            # ---- crítico (amostras por passo, estado global) ----
            permc = torch.randperm(n_t, device=self.device)
            for s in range(0, n_t, mb):
                idx = permc[s : s + mb]
                v = self.critic(c_gs[idx]).squeeze(-1)
                v_old = c_vold[idx]
                ret = c_ret[idx]
                # Value clipping (PPO): clipa V em torno de V_old ± ε
                v_clipped = v_old + torch.clamp(v - v_old, -clip, clip)
                v_loss = torch.max(
                    F.mse_loss(v, ret, reduction="none"),
                    F.mse_loss(v_clipped, ret, reduction="none"),
                )
                critic_loss = self.config.VALUE_LOSS_COEF * 0.5 * v_loss.mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.MAX_GRAD_NORM)
                self.critic_optimizer.step()
                total_critic_loss += critic_loss.item()

            # Early-stop por KL: protege contra over-update / colapso da política
            if target_kl is not None and kl_n > 0 and (kl_sum / kl_n) > 1.5 * target_kl:
                break

        self._episodes = []
        denom = max(1, n_updates)
        return total_actor_loss / denom, total_critic_loss / max(1, self.config.PPO_EPOCHS)


def run(config=None, num_sessions=1, record_video=True):
    """Treina o MAPPO canônico no WarehouseEnv simples e salva métricas/plot/vídeo."""
    config = config or MAPPOConfig()
    total_episodes = config.EPISODES_PER_SESSION * num_sessions
    rollout_eps = max(1, getattr(config, "ROLLOUT_EPISODES", 1))

    print("=" * 80)
    print("🤖 TREINAMENTO MAPPO — ator compartilhado + crítico centralizado (CTDE)")
    print("=" * 80)
    print(
        f"   • Episódios: {total_episodes} | Max steps/ep: {config.MAX_STEPS} | "
        f"Rollout: {rollout_eps} ep/update"
    )

    base_dir = Path(config.BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    env = WarehouseEnv(config=config)
    env.reset()
    n_agents = env.num_robots
    state_dim = len(env._get_observation_for_robot(0))
    global_state_dim = len(env._get_global_state())
    action_dim = env.num_actions

    controller = MAPPOController(n_agents, state_dim, action_dim, global_state_dim, config)

    metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
    }

    for ep in tqdm(range(total_episodes), desc="MAPPO"):
        env.reset()
        controller.start_episode()
        global_state = env._get_global_state()
        ep_reward = 0.0
        step = 0
        terminated = False
        info = env._get_info()
        for step in range(config.MAX_STEPS):
            local_obs = [env._get_observation_for_robot(i) for i in range(n_agents)]
            actions = controller.select_actions(local_obs, global_state, training=True)
            _, rewards, terminated, truncated, info = env.step(actions)
            controller.record_reward(sum(rewards))
            ep_reward += sum(rewards)
            global_state = env._get_global_state()  # s_{t+1}
            if terminated or truncated:
                break

        # global_state agora é o estado final s_T; bootstrapa só se NÃO terminou de fato
        controller.finish_episode(global_state, terminated)

        if (ep + 1) % rollout_eps == 0 or (ep + 1) == total_episodes:
            controller.update()

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
                f"Entregas: {np.mean(rd):.2f}/{env.num_boxes}"
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
        # Avaliação determinística (argmax), obs por-robô + estado global compartilhado.
        gs = env._get_global_state()
        local_obs = [env._get_observation_for_robot(i) for i in range(n_agents)]
        return controller.select_actions(local_obs, gs, training=False)

    video_path = None
    if record_video:
        results_dir = base_dir / "final_results"
        os.makedirs(results_dir, exist_ok=True)
        video_path = record_policy_video(config, _act, results_dir)

    return controller, metrics, video_path
