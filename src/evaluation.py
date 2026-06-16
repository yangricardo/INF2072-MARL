"""Avaliação (gravação de vídeo) e gráficos consolidados.

Extraído de: Código/Ambiente e Execução IDQN - Versão 1.3.0.py
(evaluate_and_record_video, plot_consolidated_results).

A seleção de ações na avaliação usa ``agent.select_action(obs, training=False)``,
funcionando de forma idêntica para IDQN (greedy, sem exploração) e para o
baseline aleatório.
"""

from concurrent.futures import ThreadPoolExecutor
from functools import cached_property

import imageio
import matplotlib.pyplot as plt
import numpy as np

from .environment import WarehouseEnv

# Shared background pool for video encoding and plot saving (I/O-bound)
# Call _bg_pool.shutdown(wait=True) at application exit to ensure all I/O completes.
_bg_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="eval_io")


def evaluate_and_record_video(agents, config, save_dir, num_episodes=1):
    """Avalia os agentes (sem exploração) e grava um vídeo da execução."""

    env = WarehouseEnv(config=config)
    frames = []
    episode_stats = []

    print(f"\n{'=' * 60}")
    print("🎬 AVALIAÇÃO E GERAÇÃO DE VÍDEO")
    print(f"{'=' * 60}")

    for episode in range(num_episodes):
        print(f"\n📹 Gravando episódio {episode + 1}/{num_episodes}...")

        obs, _ = env.reset()
        episode_reward = 0
        episode_deliveries = 0
        step = 0

        while True:
            try:
                frames.append(env.render_frame())
            except Exception as e:
                print(f"  ⚠️ Erro ao capturar frame: {e}")

            actions = [agent.select_action(obs, training=False) for agent in agents]
            next_obs, rewards, terminated, truncated, info = env.step(actions)

            episode_reward += sum(rewards)
            episode_deliveries = info["total_deliveries"]
            step += 1
            obs = next_obs

            if terminated or truncated:
                try:
                    frames.append(env.render_frame())
                except Exception:
                    pass
                break

        episode_stats.append(
            {
                "episode": episode,
                "reward": episode_reward,
                "deliveries": episode_deliveries,
                "steps": step,
            }
        )
        print(
            f"  ✅ Episódio {episode + 1} | Steps: {step} | "
            f"Reward: {episode_reward:.2f} | Entregas: {episode_deliveries}/{env.num_boxes}"
        )

    env.close()

    video_path = None
    if frames:
        video_path = save_dir / "robot_movement.mp4"
        _frames_snap = list(frames)
        _vpath = video_path

        def _save_video(path, frms):
            try:
                print(f"\n💾 Salvando vídeo com {len(frms)} frames...")
                imageio.mimsave(path, frms, fps=4)
                print(f"✅ Vídeo salvo em: {path}")
            except Exception as e:
                print(f"⚠️ Erro ao salvar vídeo: {e}")

        # Encode and write video in background — does not block the caller
        _bg_pool.submit(_save_video, _vpath, _frames_snap)

    return video_path, episode_stats


def record_policy_video(config, act_fn, save_dir, num_episodes=1):
    """Grava um vídeo executando uma política genérica.

    ``act_fn(env, obs) -> list[int]`` devolve as ações conjuntas de um passo;
    encapsula como cada algoritmo escolhe ações (obs compartilhada ou por-robô),
    servindo a IDQN/VDN/QMIX/MAPPO/HATRPO uniformemente.
    """
    env = WarehouseEnv(config=config)
    frames = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        while True:
            try:
                frames.append(env.render_frame())
            except Exception as e:
                print(f"  ⚠️ Erro ao capturar frame: {e}")
            actions = act_fn(env, obs)
            obs, _, terminated, truncated, _ = env.step(actions)
            if terminated or truncated:
                try:
                    frames.append(env.render_frame())
                except Exception:
                    pass
                break
    env.close()

    video_path = None
    if frames:
        video_path = save_dir / "robot_movement.mp4"
        _frames_snap = list(frames)
        _vpath = video_path

        def _save_video(path, frms):
            try:
                print(f"\n💾 Salvando vídeo com {len(frms)} frames...")
                imageio.mimsave(path, frms, fps=4)
                print(f"✅ Vídeo salvo em: {path}")
            except Exception as e:
                print(f"⚠️ Erro ao salvar vídeo: {e}")

        # Encode and write video in background — does not block the caller
        _bg_pool.submit(_save_video, _vpath, _frames_snap)
    return video_path


def plot_consolidated_results(metrics, save_dir):
    """Gera o painel 2x3 de métricas consolidadas."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor("white")

    # 1. Recompensa
    axes[0, 0].plot(metrics["episode_rewards"], alpha=0.3, linewidth=0.5)
    if len(metrics["episode_rewards"]) >= 100:
        moving_avg = np.convolve(
            metrics["episode_rewards"], np.ones(100) / 100, mode="valid"
        )
        axes[0, 0].plot(
            range(99, len(metrics["episode_rewards"])), moving_avg, "r-", linewidth=2
        )
    axes[0, 0].set_title("Recompensa Total Consolidada")
    axes[0, 0].set_xlabel("Episódio")
    axes[0, 0].set_ylabel("Recompensa")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor("white")

    # 2. Entregas
    axes[0, 1].plot(
        metrics["episode_deliveries"], alpha=0.3, linewidth=0.5, color="green"
    )
    if len(metrics["episode_deliveries"]) >= 100:
        moving_avg_del = np.convolve(
            metrics["episode_deliveries"], np.ones(100) / 100, mode="valid"
        )
        axes[0, 1].plot(
            range(99, len(metrics["episode_deliveries"])),
            moving_avg_del,
            "r-",
            linewidth=2,
        )
    axes[0, 1].axhline(y=4, color="g", linestyle="--", label="Meta (4 entregas)")
    axes[0, 1].set_title("Entregas Consolidadas")
    axes[0, 1].set_xlabel("Episódio")
    axes[0, 1].set_ylabel("Entregas")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_facecolor("white")

    # 3. Steps
    axes[0, 2].plot(metrics["episode_steps"], color="orange", alpha=0.5)
    axes[0, 2].set_title("Steps por Episódio")
    axes[0, 2].set_xlabel("Episódio")
    axes[0, 2].set_ylabel("Steps")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_facecolor("white")

    # 4. Taxa de sucesso
    axes[1, 0].plot(metrics["success_rates"], color="purple", alpha=0.5)
    if len(metrics["success_rates"]) >= 100:
        moving_avg_success = np.convolve(
            metrics["success_rates"], np.ones(100) / 100, mode="valid"
        )
        axes[1, 0].plot(
            range(99, len(metrics["success_rates"])),
            moving_avg_success,
            "r-",
            linewidth=2,
        )
    axes[1, 0].axhline(y=0.95, color="g", linestyle="--", label="Meta 95%")
    axes[1, 0].set_title("Taxa de Sucesso Consolidada")
    axes[1, 0].set_xlabel("Episódio")
    axes[1, 0].set_ylabel("Taxa")
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_facecolor("white")

    # 5. Colisões
    axes[1, 1].plot(metrics["collisions"], color="red", alpha=0.5)
    if len(metrics["collisions"]) >= 100:
        moving_avg_coll = np.convolve(
            metrics["collisions"], np.ones(100) / 100, mode="valid"
        )
        axes[1, 1].plot(
            range(99, len(metrics["collisions"])), moving_avg_coll, "r-", linewidth=2
        )
    axes[1, 1].set_title("Colisões Consolidadas")
    axes[1, 1].set_xlabel("Episódio")
    axes[1, 1].set_ylabel("Colisões")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_facecolor("white")

    # 6. Resumo
    axes[1, 2].axis("off")
    final_stats = {
        "Total Episódios": len(metrics["episode_rewards"]),
        "Melhor Recompensa": f"{np.max(metrics['episode_rewards']):.2f}",
        "Média Entregas": f"{np.mean(metrics['episode_deliveries'][-100:]):.2f}",
        "Taxa Sucesso Final": f"{metrics['success_rates'][-1]:.1%}",
    }

    axes[1, 2].text(0.1, 0.9, "📊 RESUMO FINAL", fontsize=12, fontweight="bold")
    y = 0.8
    for key, value in final_stats.items():
        axes[1, 2].text(0.1, y, f"{key}: {value}", fontsize=10)
        y -= 0.1

    plt.tight_layout()
    fig_snap = fig  # capture reference before potential GC

    def _save_plot(f, path):
        try:
            f.savefig(path, dpi=150, facecolor="white")
            plt.close(f)
        except Exception as e:
            print(f"⚠️ Erro ao salvar gráfico: {e}")

    # Write PNG in background — matplotlib savefig is I/O-bound after rendering
    _bg_pool.submit(_save_plot, fig_snap, save_dir / "consolidated_results.png")
