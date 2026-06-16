"""Loop de treino multi-sessão, reutilizável por qualquer agente.

Extraído de: Código/Ambiente e Execução IDQN - Versão 1.3.0.py
(train_session, run_multi_session_training). Usado por IDQN e Random.

``run_training`` recebe uma *fábrica de agentes* (``agent_class``) com assinatura
``(state_dim, action_dim, agent_id, config)`` — tanto ``IDQNAgent`` quanto
``RandomAgent`` a satisfazem, então o mesmo loop serve para ambos. Agentes sem
``policy_net`` (ex.: baseline aleatório) simplesmente não salvam pesos.
"""

import os
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .environment import WarehouseEnv
from .evaluation import evaluate_and_record_video, plot_consolidated_results


def _should_optimize(agent, config):
    """Verifica se optimize() fará trabalho útil antes de submeter ao pool.

    Apenas verifica pré-condições (buffer size e learning starts). Não replica
    TRAIN_FREQ — o método optimize() do agente gerencia sua própria frequência
    internamente (incrementa learning_steps e verifica o módulo), como no
    código original (Ambiente e Execução IDQN - Versão 1.3.0.py).
    """
    if len(agent.memory) < config.BATCH_SIZE:
        return False
    if agent.steps_done < config.LEARNING_STARTS:
        return False
    return True


def train_session(session_dir, agents, config, session_id=1, start_episode=0):
    """Executa uma sessão de treinamento com tratamento de erros robusto."""

    env = WarehouseEnv(config=config)
    total_episodes = start_episode + config.EPISODES_PER_SESSION

    try:
        os.makedirs(session_dir, exist_ok=True)
        metrics_dir = session_dir / "metrics"
        models_dir = session_dir / "models"
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    except Exception as e:
        print(f"⚠️ Erro ao criar diretórios: {e}")
        session_dir = Path("./temp_training")
        metrics_dir = session_dir / "metrics"
        models_dir = session_dir / "models"
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)

    metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
        "distance_traveled": [],
    }

    best_reward = -float("inf")
    # Background I/O pool — checkpoints and CSV writes don't block the train loop
    _io_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="io")
    # Persistent optimize pool — created once, reused across all steps
    _optimize_pool = ThreadPoolExecutor(max_workers=len(agents), thread_name_prefix="opt")

    print(f"\n🚀 Iniciando sessão {session_id:04d}")
    print(f"   Episódios: {start_episode} → {total_episodes}")

    for episode in tqdm(
        range(start_episode, total_episodes),
        desc=f"Sessão {session_id:04d}",
        total=total_episodes - start_episode,
    ):
        obs, info = env.reset()
        episode_reward = 0
        episode_collisions = 0
        step = 0

        for step in range(config.MAX_STEPS):
            # select_action serial — forward pass de MLP em CPU é < 1ms,
            # overhead de ThreadPoolExecutor supera qualquer ganho e steps_done
            # fica determinístico (essencial para epsilon e beta annealing)
            actions = [agent.select_action(obs) for agent in agents]
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            for i, agent in enumerate(agents):
                agent.remember(
                    obs, actions[i], rewards[i], next_obs, terminated or truncated
                )
                episode_reward += rewards[i]

            episode_collisions = info["collisions"]

            # Optimize each agent in parallel — only submit when there's real work
            futures = {}
            for agent in agents:
                if _should_optimize(agent, config):
                    futures[agent] = _optimize_pool.submit(agent.optimize)
            if futures:
                wait(list(futures.values()))

            obs = next_obs

            if terminated or truncated:
                break

        for agent in agents:
            agent.total_episodes = episode + 1

        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_deliveries"].append(info["total_deliveries"])
        metrics["episode_steps"].append(step + 1)
        metrics["success_rates"].append(info["success_rate"])
        metrics["collisions"].append(episode_collisions)
        metrics["distance_traveled"].append(sum(info["distance_traveled"]))

        if episode_reward > best_reward:
            best_reward = episode_reward
            for i, agent in enumerate(agents):
                if not hasattr(agent, "policy_net"):
                    continue
                try:
                    torch.save(
                        agent.policy_net.state_dict(),
                        models_dir / f"best_agent_{i}_best.pth",
                    )
                except Exception:
                    pass

        if config.SAVE_CHECKPOINTS and (episode + 1) % config.SAVE_CHECKPOINT_EVERY == 0:
            # Snapshot metrics and agent states now; write to disk in background
            _ckpt_dir = session_dir / f"checkpoint_{episode + 1}"
            _ckpt_ep = episode + 1
            _ckpt_rewards = metrics["episode_rewards"][:]
            _ckpt_deliveries = metrics["episode_deliveries"][:]
            _agents_snap = agents  # agents are not replaced between episodes

            def _write_checkpoint(ckpt_dir, ep, rewards, deliveries, ag, mdir):
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    for agent in ag:
                        agent.save_checkpoint(ckpt_dir)
                    pd.DataFrame(
                        {
                            "episode": range(len(rewards)),
                            "reward": rewards,
                            "deliveries": deliveries,
                        }
                    ).to_csv(mdir / f"metrics_checkpoint_{ep}.csv", index=False)
                except Exception as e:
                    print(f"\n  ⚠️ Erro ao salvar checkpoint: {e}")

            _io_pool.submit(
                _write_checkpoint,
                _ckpt_dir, _ckpt_ep, _ckpt_rewards, _ckpt_deliveries,
                _agents_snap, metrics_dir,
            )

        if (episode + 1) % 100 == 0:
            recent_rewards = metrics["episode_rewards"][-100:]
            recent_deliveries = metrics["episode_deliveries"][-100:]
            epsilon = agents[0].get_epsilon()

            print(
                f"Sessão {session_id:04d} | Ep {episode + 1:5d} | "
                f"Reward: {np.mean(recent_rewards):7.2f} | "
                f"Entregas: {np.mean(recent_deliveries):.2f} | "
                f"ε: {epsilon:.3f}"
            )

    env.close()
    _io_pool.shutdown(wait=True)  # Flush pending checkpoint writes before returning
    _optimize_pool.shutdown(wait=True)  # N9: Release optimize threads

    try:
        df = pd.DataFrame(
            {
                "episode": range(
                    start_episode, len(metrics["episode_rewards"]) + start_episode
                ),
                "reward": metrics["episode_rewards"],
                "deliveries": metrics["episode_deliveries"],
                "steps": metrics["episode_steps"],
                "success_rate": metrics["success_rates"],
                "collisions": metrics["collisions"],
            }
        )
        df.to_csv(metrics_dir / "training_metrics.csv", index=False)
    except Exception as e:
        print(f"⚠️ Erro ao salvar métricas: {e}")

    return metrics


def run_training(agent_class, config, num_sessions=1, record_video=True):
    """Executa múltiplas sessões de treino para a classe de agente informada.

    ``agent_class`` é qualquer classe com assinatura
    ``(state_dim, action_dim, agent_id, config)`` — ex.: ``IDQNAgent`` ou
    ``RandomAgent``. Dois agentes são instanciados (um por robô).
    """

    print("=" * 80)
    print(f"🏭 TREINAMENTO MULTI-SESSÃO — {agent_class.__name__} — 2 ROBÔS")
    print("=" * 80)
    print("\n📋 Configuração:")
    print(f"   • Episódios por sessão: {config.EPISODES_PER_SESSION}")
    print(f"   • Total de sessões: {num_sessions}")
    print(f"   • Total de episódios: {config.EPISODES_PER_SESSION * num_sessions}")
    print("=" * 80)

    base_dir = Path(config.BASE_DIR)
    os.makedirs(base_dir, exist_ok=True)

    # Inicializa ambiente para obter dimensões
    env = WarehouseEnv(config=config)
    sample_obs, _ = env.reset()
    state_dim = len(sample_obs)
    action_dim = env.num_actions
    env.close()

    agents = [agent_class(state_dim, action_dim, i, config) for i in range(2)]

    # N11: Configura o total de steps de treino para annealing do beta no PER
    total_train_steps = config.EPISODES_PER_SESSION * num_sessions * config.MAX_STEPS
    for agent in agents:
        if hasattr(agent, 'set_total_train_steps'):
            agent.set_total_train_steps(total_train_steps)

    all_metrics = []
    total_episodes_done = 0
    last_session_dir = None

    for session_num in range(1, num_sessions + 1):
        session_dir = base_dir / f"session_{session_num:04d}"

        if session_num > 1 and last_session_dir:
            checkpoint_dir = last_session_dir / f"checkpoint_{config.EPISODES_PER_SESSION}"
            if checkpoint_dir.exists():
                print("\n📂 Carregando checkpoint da sessão anterior...")
                for i, agent in enumerate(agents):
                    agent.load_checkpoint(checkpoint_dir, i)

        session_metrics = train_session(
            session_dir,
            agents,
            config,
            session_id=session_num,
            start_episode=total_episodes_done,
        )

        all_metrics.append(session_metrics)
        total_episodes_done += config.EPISODES_PER_SESSION
        last_session_dir = session_dir

        try:
            models_dir = session_dir / "models"
            for i, agent in enumerate(agents):
                if hasattr(agent, "policy_net"):
                    torch.save(
                        agent.policy_net.state_dict(),
                        models_dir / f"agent_{i}_final.pth",
                    )
        except Exception as e:
            print(f"⚠️ Erro ao salvar modelo final: {e}")

        print(f"\n✅ Sessão {session_num:04d} concluída!")
        print(f"   Total de episódios treinados: {total_episodes_done}")

    # Consolidar métricas
    consolidated_metrics = {
        "episode_rewards": [],
        "episode_deliveries": [],
        "episode_steps": [],
        "success_rates": [],
        "collisions": [],
        "distance_traveled": [],
    }
    for metrics in all_metrics:
        for key in consolidated_metrics:
            consolidated_metrics[key].extend(metrics[key])

    consolidated_dir = base_dir / "consolidated_results"
    os.makedirs(consolidated_dir, exist_ok=True)

    pd.DataFrame(
        {
            "episode": range(1, len(consolidated_metrics["episode_rewards"]) + 1),
            "reward": consolidated_metrics["episode_rewards"],
            "deliveries": consolidated_metrics["episode_deliveries"],
            "steps": consolidated_metrics["episode_steps"],
            "success_rate": consolidated_metrics["success_rates"],
            "collisions": consolidated_metrics["collisions"],
            "distance_traveled": consolidated_metrics["distance_traveled"],
        }
    ).to_csv(consolidated_dir / "consolidated_metrics.csv", index=False)

    plot_consolidated_results(consolidated_metrics, consolidated_dir)
    print(f"\n📁 Resultados consolidados salvos em: {consolidated_dir}")

    # Gerar vídeo da política final
    video_path = None
    if record_video:
        results_dir = base_dir / "final_results"
        os.makedirs(results_dir, exist_ok=True)
        video_path, _ = evaluate_and_record_video(
            agents, config, results_dir, num_episodes=1
        )

    return agents, consolidated_metrics, video_path
