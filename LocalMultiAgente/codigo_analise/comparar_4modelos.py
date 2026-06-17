#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativo de 4 MODELOS para a apresentação (INF2072 / PUC-DI).

Modelos (exatamente estes 4, conforme pedido):
    Random | IDQN (melhor) | MAPPO (corrigido) | VDN v6 (curriculum)

Métricas (média móvel de 100 episódios):
    taxa de sucesso | colisões | entregas | falhas

Gera dois conjuntos de gráficos em comparativos/4modelos/:
  1) COMPARATIVO  — as 4 métricas com os 4 modelos coloridos juntos.
  2) DESTAQUE     — para cada métrica, 4 versões (uma por modelo): o modelo em
                    destaque fica colorido e os outros três ficam cinza.
                    (4 métricas × 4 modelos = 16 gráficos)

DECISÕES DE COMPARABILIDADE (importantes e honestas):
  • taxa_sucesso é RECALCULADA como entregas/8 para TODOS os modelos, para ficar
    na mesma régua. O VDN v6 usou curriculum (2→4→6→8 caixas), então a coluna
    'taxa_sucesso' original dele é sobre "caixas ativas" e NÃO seria comparável.
  • RECOMPENSA fica de fora: o VDN v6 usa escala diferente (entrega=80 vs 25),
    então recompensas não são comparáveis. As 4 métricas pedidas não dependem
    disso (são contagens físicas).
  • Colisões/Falhas refletem o ambiente estocástico (taxa de falha de movimento
    difere entre modelos: IDQN 20%, VDN v6 10%). Anotado nos gráficos.

Uso:
    python comparar_4modelos.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RESULTS = BASE / "resultados"
OUT_DIR = BASE / "comparativos" / "4modelos"
DEST_DIR = OUT_DIR / "destaque"

WINDOW = 100
GRAY = "#b8b8b8"

# (label, caminho do csv relativo a resultados/, cor, slug p/ nome de arquivo)
MODELS = [
    ("Random",  "random/metricas.csv",                                   "#2ca02c", "random"),
    ("IDQN",    "idqn/metricas.csv",                                     "#1f77b4", "idqn"),
    ("MAPPO",   "mappo_corrigido/metricas.csv",                          "#ff7f0e", "mappo"),
    ("VDN v6",  "vdn_results_v6_20260615_213758/metricas_treinamento_vdn.csv", "#d62728", "vdn"),
]

# Métricas: (chave, função p/ extrair série do df, ylabel, título, hline, nota)
def _entregas(df):      return df["entregas"].to_numpy(dtype=float)
def _sucesso(df):       return df["entregas"].to_numpy(dtype=float) / 8.0  # entregas/8 p/ todos
def _colisoes(df):      return df["colisoes"].to_numpy(dtype=float)
def _falhas(df):        return (df["falha_r1"].to_numpy(dtype=float)
                                + df["falha_r2"].to_numpy(dtype=float))

NOTE_ENV = ("Ambiente estocástico: taxa de falha de movimento difere por modelo "
            "(IDQN 20%, VDN v6 10%). VDN v6 usou curriculum 2→4→6→8 caixas.")

METRICS = [
    ("sucesso",  _sucesso,  "Taxa de sucesso (entregas / 8)",
     "Taxa de Sucesso por Episódio", 1.0,
     "Taxa de sucesso = entregas/8 (mesma régua p/ todos). VDN v6 usou curriculum."),
    ("colisoes", _colisoes, "Colisões",
     "Colisões por Episódio", None, NOTE_ENV),
    ("entregas", _entregas, "Entregas (de 8)",
     "Entregas por Episódio", 8.0,
     "Meta = 8 caixas. VDN v6 usou curriculum (entregas limitadas às caixas ativas no início)."),
    ("falhas",   _falhas,   "Falhas de movimento (R1+R2)",
     "Falhas por Episódio", None, NOTE_ENV),
]


def moving_avg(series, window=WINDOW):
    series = np.asarray(series, dtype=float)
    if len(series) < window:
        return np.arange(1, len(series) + 1), series
    ma = np.convolve(series, np.ones(window) / window, mode="valid")
    return np.arange(window, len(series) + 1), ma


def load_models():
    data = {}
    for label, rel, color, slug in MODELS:
        csv = RESULTS / rel
        if not csv.exists():
            print(f"  [AVISO] CSV não encontrado p/ {label}: {csv} — pulando.")
            continue
        df = pd.read_csv(csv)
        data[label] = dict(df=df, color=color, slug=slug)
        print(f"  [OK] {label:8} <- {rel}  ({len(df)} episódios)")
    return data


def _decorate(ax, ylabel, title, hline, note):
    ax.set_xlabel("Episódio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if hline is not None:
        ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    if note:
        ax.text(0.5, -0.13, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, style="italic", color="dimgray", wrap=True)


def plot_comparativo(data, key, getter, ylabel, title, hline, note):
    fig, ax = plt.subplots(figsize=(12, 7))
    for label, d in data.items():
        x, ma = moving_avg(getter(d["df"]))
        ax.plot(x, ma, color=d["color"], linewidth=2.2, label=label)
    _decorate(ax, ylabel, f"{title} — Comparativo (3000 ep, média móvel 100)", hline, note)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()
    out = OUT_DIR / f"comp_{key}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.relative_to(BASE)}")


def plot_destaque(data, key, getter, ylabel, title, hline, note, hl_label):
    fig, ax = plt.subplots(figsize=(12, 7))
    # primeiro os cinzas (fundo), depois o destaque (frente)
    for label, d in data.items():
        if label == hl_label:
            continue
        x, ma = moving_avg(getter(d["df"]))
        ax.plot(x, ma, color=GRAY, linewidth=1.3, alpha=0.8, label=label, zorder=1)
    d = data[hl_label]
    x, ma = moving_avg(getter(d["df"]))
    ax.plot(x, ma, color=d["color"], linewidth=3.0, label=hl_label, zorder=5)
    _decorate(ax, ylabel, f"{title} — destaque {hl_label} (3000 ep, MM 100)", hline, note)
    # legenda: destaque em negrito
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(handles, labels, fontsize=11, loc="best")
    for t in leg.get_texts():
        if t.get_text() == hl_label:
            t.set_fontweight("bold")
    fig.tight_layout()
    out = DEST_DIR / f"destaque_{key}_{d['slug']}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.relative_to(BASE)}")


def plot_painel(data, filename="comp_painel.png"):
    """Painel 2x2 com as 4 métricas (4 modelos coloridos) num único slide."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for ax, (key, getter, ylabel, title, hline, _note) in zip(axes.flat, METRICS):
        for label, d in data.items():
            x, ma = moving_avg(getter(d["df"]))
            ax.plot(x, ma, color=d["color"], linewidth=2.0, label=label)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episódio"); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=10, loc="best")
    fig.suptitle("Comparativo dos 4 modelos — média móvel de 100 episódios (3000 ep)",
                 fontsize=16, fontweight="bold")
    fig.text(0.5, 0.005,
             "Taxa de sucesso = entregas/8 (mesma régua). VDN v6 usou curriculum 2→4→6→8 e "
             "failure prob 10% (IDQN 20%) → colisões/falhas não comparáveis 1:1.",
             ha="center", fontsize=9, style="italic", color="dimgray")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.relative_to(BASE)}")


def main():
    print("=" * 72)
    print("COMPARATIVO 4 MODELOS — Random | IDQN | MAPPO (corrig.) | VDN v6")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    data = load_models()
    if len(data) < 2:
        print("ERRO: poucos modelos carregados.")
        return 1

    print("\n[1/2] Gráficos comparativos (4 modelos juntos)...")
    for key, getter, ylabel, title, hline, note in METRICS:
        plot_comparativo(data, key, getter, ylabel, title, hline, note)
    plot_painel(data)

    print("\n[2/2] Gráficos com destaque (16: 4 métricas × 4 modelos)...")
    for key, getter, ylabel, title, hline, note in METRICS:
        for hl_label in data.keys():
            plot_destaque(data, key, getter, ylabel, title, hline, note, hl_label)

    # resumo final (últimos 100 ep) p/ conferência
    print("\nResumo (média dos últimos 100 episódios):")
    print(f"  {'modelo':8} {'entregas/8':>11} {'sucesso%':>9} {'colisões':>9} {'falhas':>8}")
    for label, d in data.items():
        tail = d["df"].tail(WINDOW)
        ent = tail["entregas"].mean()
        suc = ent / 8.0 * 100
        col = tail["colisoes"].mean()
        fal = (tail["falha_r1"] + tail["falha_r2"]).mean()
        print(f"  {label:8} {ent:11.2f} {suc:8.1f}% {col:9.1f} {fal:8.1f}")

    print(f"\nConcluído. Saídas em: {OUT_DIR.relative_to(BASE)}/ e {DEST_DIR.relative_to(BASE)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
