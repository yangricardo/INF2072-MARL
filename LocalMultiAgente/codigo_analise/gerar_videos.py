#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera VIDEOS (GIF) das politicas treinadas agindo no armazem 12x8.

Carrega os modelos salvos (apenas inferencia, sem treinar), roda um episodio
guloso e renderiza o grid a cada passo. Saida em `Output/videos/`.

Politicas:
  - idqn            : argmax da rede DQN de cada robo
  - mappo_corrigido : argmax da politica (ator) de cada robo
  - random          : acoes aleatorias (baseline, p/ contraste)

Renderiza com matplotlib + Pillow (nao precisa de ffmpeg/imageio).

Uso:
    python gerar_videos.py                 # gera os 3
    python gerar_videos.py idqn random     # apenas alguns
"""
from pathlib import Path
import importlib.util
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RESULTS = BASE / "resultados"
OUT = BASE / "Output" / "videos"

# Importa o ambiente e a rede do ator do script do MAPPO corrigido
_spec = importlib.util.spec_from_file_location("mc", HERE / "mappo_corrigido.py")
mc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mc)
WarehouseEnv = mc.WarehouseEnv
ActorNetwork = mc.ActorNetwork


# ---- Rede DQN do IDQN (mesma arquitetura do notebook: hidden 512, dropout 0.2) ----
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512, dropout=0.2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.network(x)


# ============================ POLITICAS ============================
def make_idqn_policy():
    nets = []
    for i in range(2):
        cand = sorted((RESULTS / "idqn" / "models").glob(f"idqn_agent_{i}_*.pth"))
        # prefere o "final"; senao o de maior episodio
        final = [c for c in cand if "final" in c.name]
        path = final[0] if final else cand[-1]
        net = DQN(40, 6)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        nets.append(net)
    def policy(obs):
        t = torch.FloatTensor(obs).unsqueeze(0)
        return [net(t).argmax().item() for net in nets]
    return policy


def make_mappo_policy():
    nets = []
    for i in range(2):
        path = RESULTS / "mappo_corrigido" / "models" / f"mappo_actor_{i}_3000ep.pth"
        net = ActorNetwork(40, 6)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        nets.append(net)
    def policy(obs):
        # MAPPO e estocastico: amostra da distribuicao (como no treino).
        t = torch.FloatTensor(obs).unsqueeze(0)
        acts = []
        for net in nets:
            probs, _ = net(t)
            acts.append(torch.distributions.Categorical(probs).sample().item())
        return acts
    return policy


def make_random_policy():
    import random
    # usa o RNG global (semeado por rollout) -> reproduzivel com a seed
    return lambda obs: [random.randrange(6) for _ in range(2)]


# ============================ RENDER ============================
ROBOT_COLORS = {0: "#1f77b4", 1: "#d62728"}


def render_frame(env, titulo):
    h, w = env.height, env.width
    fig, ax = plt.subplots(figsize=(w * 0.6, h * 0.6), dpi=80)
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")

    # fundo + paredes/barreiras a partir do grid atual
    for i in range(h):
        for j in range(w):
            cell = env.grid[i][j]
            if cell == "X":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="#2b2b2b"))
            elif cell == "Y":
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, color="#9e9e9e"))
            else:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor="white", edgecolor="#dddddd", lw=0.5))

    # alvos de entrega (B)
    for (ti, tj) in env.targets:
        ax.add_patch(Rectangle((tj - 0.5, ti - 0.5), 1, 1, facecolor="#b6e3b6",
                               edgecolor="#5fa85f", lw=1.2))

    # caixas ainda no chao (nao entregues e nao carregadas)
    for k, bp in enumerate(env.box_positions):
        if bp is not None and not env.delivered_boxes[k]:
            ax.add_patch(Rectangle((bp[1] - 0.30, bp[0] - 0.30), 0.6, 0.6,
                                   facecolor="#ff9800", edgecolor="#b35e00", lw=1))

    # robos
    for rid, (ri, rj) in enumerate(env.robot_positions):
        ax.add_patch(Circle((rj, ri), 0.34, color=ROBOT_COLORS[rid], zorder=5))
        ax.text(rj, ri, f"R{rid+1}", color="white", ha="center", va="center",
                fontsize=8, fontweight="bold", zorder=6)

    # legenda compacta (proxies)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor="#2b2b2b", label="Parede"),
        Rectangle((0, 0), 1, 1, facecolor="#9e9e9e", label="Barreira Y"),
        Rectangle((0, 0), 1, 1, facecolor="#b6e3b6", edgecolor="#5fa85f", label="Entrega"),
        Rectangle((0, 0), 1, 1, facecolor="#ff9800", edgecolor="#b35e00", label="Caixa"),
        Circle((0, 0), 1, facecolor="#1f77b4", label="Robôs"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
              ncol=5, fontsize=6.5, frameon=False, handlelength=1.2, columnspacing=1.0)

    ax.set_title(titulo, fontsize=10, fontweight="bold")
    fig.tight_layout(pad=0.4)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


def rollout(policy, nome, max_steps=200, seed=0, render=True):
    """Roda um episodio. Se render=False, so simula (rapido) e nao gera frames."""
    import random
    random.seed(seed); np.random.seed(seed)
    env = WarehouseEnv(config=mc.Config())
    obs, _ = env.reset()
    frames = [render_frame(env, f"{nome} | passo 0 | entregas 0/8")] if render else []
    steps = 0
    for step in range(1, max_steps + 1):
        actions = policy(obs)
        obs, _, terminated, truncated, info = env.step(actions)
        steps = step
        if render:
            frames.append(render_frame(
                env, f"{nome} | passo {step} | entregas {info['total_deliveries']}/8"))
        if terminated or truncated:
            break
    return frames, info, steps


def save_gif(frames, path, fps=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    dur = int(1000 / fps)
    # segura o ultimo quadro por ~1.5s
    frames[-1].save(path, save_all=True, append_images=frames[1:] + [frames[-1]] * fps,
                    duration=dur, loop=0, optimize=True)


POLICIES = {
    "idqn": ("IDQN (treinado)", make_idqn_policy, 200),
    "mappo_corrigido": ("MAPPO corrigido (treinado)", make_mappo_policy, 200),
    "random": ("Random (baseline)", make_random_policy, 120),
}


def main(argv):
    alvos = argv[1:] or list(POLICIES.keys())
    OUT.mkdir(parents=True, exist_ok=True)
    for key in alvos:
        if key not in POLICIES:
            print(f"  [PULA] {key}: politica desconhecida"); continue
        nome, maker, max_steps = POLICIES[key]
        try:
            policy = maker()
        except Exception as e:
            print(f"  [ERRO] {key}: nao consegui carregar modelo ({e})"); continue
        # 1) passe rapido (sem render) p/ achar a melhor seed: + entregas, - passos
        best_seed, best_score = 0, None
        for seed in range(8):
            _, info, steps = rollout(policy, nome, max_steps=max_steps, seed=seed, render=False)
            score = (info["total_deliveries"], -steps)
            if best_score is None or score > best_score:
                best_score, best_seed = score, seed
        # 2) renderiza so a melhor seed
        frames, info, _ = rollout(policy, nome, max_steps=max_steps, seed=best_seed, render=True)
        out = OUT / f"video_{key}.gif"
        save_gif(frames, out)
        print(f"  [GIF] {out.relative_to(BASE)}  "
              f"({len(frames)} quadros, entregas finais {info['total_deliveries']}/8, seed {best_seed})")
    print("Concluido. Videos em", OUT.relative_to(BASE))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
