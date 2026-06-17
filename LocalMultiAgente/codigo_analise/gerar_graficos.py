#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera os graficos POR-MODELO de forma padronizada e consistente, a partir dos
CSVs reais em `resultados/<modelo>/metricas.csv`.

Todos os graficos saem no MESMO estilo: serie bruta (fraca) + media movel de 100
episodios (linha forte). Saida em `resultados/<modelo>/graficos/`.

Isso substitui os PNGs originais (que tinham estilos/nomes inconsistentes e
estao preservados em `resultados/<modelo>/graficos_originais/`).

Uso:
    python gerar_graficos.py            # todos os modelos
    python gerar_graficos.py idqn vdn   # apenas alguns
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent          # .../codigo_analise
BASE = HERE.parent                               # .../LocalMultiAgente
RESULTS = BASE / "resultados"
WINDOW = 100

# Nome bonito de cada modelo (para titulos)
NOMES = {
    "random": "Random (baseline)",
    "idqn": "IDQN",
    "vdn": "VDN",
    "mappo_original": "MAPPO (original, com bug)",
    "mappo_corrigido": "MAPPO (corrigido)",
}

# Apelidos de colunas divergentes -> nome canonico
ALIASES = {"distancia": "distancia_percorrida"}

# (coluna, titulo, rotulo_y, cor_serie, [(hline_y, hline_cor, hline_label)])
METRICAS = [
    ("recompensa",   "Recompensa por Episodio", "Recompensa", "tab:blue",   None),
    ("entregas",     "Entregas por Episodio",   "Entregas (de 8)", "tab:green",
        [(8, "orange", "Meta (8 caixas)")]),
    ("taxa_sucesso", "Taxa de Sucesso por Episodio", "Taxa de sucesso (entregas/8)", "tab:purple",
        [(0.95, "green", "Meta 95%")]),
    ("colisoes",     "Colisoes por Episodio",   "Colisoes", "tab:red",    None),
    ("steps",        "Steps por Episodio",      "Steps", "tab:orange", None),
]


def _ma(y):
    y = np.asarray(y, dtype=float)
    if len(y) < WINDOW:
        return [], []
    ma = np.convolve(y, np.ones(WINDOW) / WINDOW, mode="valid")
    return range(WINDOW, len(y) + 1), ma


def _plot(df, col, titulo, ylabel, cor, hlines, modelo_nome, out_path, ylim=None):
    if col not in df.columns:
        return False
    y = df[col].to_numpy()
    eps = range(1, len(y) + 1)
    plt.figure(figsize=(13, 7))
    plt.plot(eps, y, color=cor, alpha=0.25, linewidth=0.8, label="Por episodio")
    x_ma, ma = _ma(y)
    if len(ma):
        plt.plot(x_ma, ma, "r-", linewidth=2.2, label=f"Media movel ({WINDOW})")
    for hy, hc, hl in (hlines or []):
        plt.axhline(y=hy, color=hc, linestyle="--", linewidth=2, label=hl)
    plt.xlabel("Episodio", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"{titulo} — {modelo_nome}", fontsize=14, fontweight="bold")
    if ylim:
        plt.ylim(ylim)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return True


def gerar_para_modelo(folder: Path) -> int:
    csv = folder / "metricas.csv"
    if not csv.exists():
        print(f"  [PULA] {folder.name}: sem metricas.csv")
        return 0
    df = pd.read_csv(csv).rename(columns=ALIASES)
    nome = NOMES.get(folder.name, folder.name)
    out = folder / "graficos"
    out.mkdir(exist_ok=True)

    n = 0
    for col, titulo, ylabel, cor, hlines in METRICAS:
        ylim = [0, 1.05] if col == "taxa_sucesso" else ([0, 9] if col == "entregas" else None)
        if _plot(df, col, titulo, ylabel, cor, hlines, nome, out / f"grafico_{col}.png", ylim):
            n += 1

    # Falhas: total + R1 + R2, com media movel do total
    if {"falha_r1", "falha_r2"}.issubset(df.columns):
        tot = (df["falha_r1"] + df["falha_r2"]).to_numpy()
        eps = range(1, len(tot) + 1)
        plt.figure(figsize=(13, 7))
        plt.plot(eps, tot, color="brown", alpha=0.25, linewidth=0.8, label="Total")
        plt.plot(eps, df["falha_r1"], color="tab:red", alpha=0.2, linewidth=0.6, label="R1")
        plt.plot(eps, df["falha_r2"], color="tab:blue", alpha=0.2, linewidth=0.6, label="R2")
        x_ma, ma = _ma(tot)
        if len(ma):
            plt.plot(x_ma, ma, "k-", linewidth=2.2, label=f"Media movel total ({WINDOW})")
        plt.xlabel("Episodio", fontsize=12)
        plt.ylabel("Falhas", fontsize=12)
        plt.title(f"Falhas por Episodio — {nome}", fontsize=14, fontweight="bold")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out / "grafico_falhas.png", dpi=150, bbox_inches="tight")
        plt.close()
        n += 1

    print(f"  [OK] {folder.name:16} -> {n} graficos em {out.relative_to(BASE)}")
    return n


def main(argv):
    alvos = argv[1:]
    folders = sorted(p for p in RESULTS.iterdir() if p.is_dir())
    if alvos:
        folders = [p for p in folders if p.name in alvos]
    if not folders:
        print("Nenhuma pasta de resultados encontrada em", RESULTS)
        return 1
    print("Gerando graficos por-modelo (com media movel) em resultados/<modelo>/graficos/ ...")
    for f in folders:
        gerar_para_modelo(f)
    print("Concluido.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
