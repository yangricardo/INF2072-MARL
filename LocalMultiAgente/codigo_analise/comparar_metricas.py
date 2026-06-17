#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparativo HONESTO dos métodos de MARL no armazém 12x8 (INF2072 / PUC-DI).

Este script lê os CSVs de métricas REAIS gravados por cada experimento
(Random, IDQN, VDN, MAPPO) e gera os gráficos comparativos e uma tabela-resumo.

POR QUE ESTE SCRIPT EXISTE
--------------------------
Os notebooks em `_SIMULADO_NAO_USAR/` NÃO usam os dados reais: eles
desenham curvas SIMULADAS (np.random.normal e dicionários digitados à mão) que
chegam a contradizer os relatórios reais (ex.: o MAPPO real entrega 0,07/8, mas
o gráfico simulado mostra "sucesso" subindo). Este script substitui aquilo por
uma comparação fiel aos CSVs de treinamento.

Definições importantes (conferidas no código dos notebooks):
- `taxa_sucesso` = total_deliveries / 8  -> FRAÇÃO das 8 caixas entregues no
  episódio. É a mesma definição nos quatro notebooks (vem de
  WarehouseEnv._get_info()['success_rate']).
- HATRPO foi implementado mas NÃO tem execução salva, então fica de fora.
- VDN rodou com MAX_STEPS=1000 (os demais com 500); por isso suas colisões
  absolutas não são diretamente comparáveis às dos outros. Isso é anotado no
  gráfico de colisões.

Uso:
    python comparar_metricas.py
Saídas (gravadas ao lado deste script):
    comparativo_recompensa.png
    comparativo_entregas.png
    comparativo_taxa_sucesso.png
    comparativo_colisoes.png
    comparativo_metricas.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # backend sem display, roda em qualquer máquina
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Estrutura: resultados/<modelo>/metricas.csv ; saídas em comparativos/
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # .../codigo_analise
BASE = HERE.parent                               # .../LocalMultiAgente
RESULTS = BASE / "resultados"
OUT_DIR = BASE / "comparativos"

WINDOW = 100  # janela da média móvel (mesma usada nos notebooks)

# Cada método: (label, subpasta em resultados/, cor)
METHODS = [
    ("Random",          "random",          "#2ca02c"),
    ("IDQN",            "idqn",            "#1f77b4"),
    ("VDN",             "vdn",             "#d62728"),
    ("VDN (ajustado)",  "vdn_ajustado",    "#17becf"),
    ("MAPPO (orig.)",   "mappo_original",  "#9467bd"),
    ("MAPPO (corrig.)", "mappo_corrigido", "#ff7f0e"),
]

# Mapeia nomes de coluna divergentes para um esquema único.
COLUMN_ALIASES = {
    "distancia": "distancia_percorrida",
}


def load_method(label: str, subfolder: str) -> pd.DataFrame | None:
    csv_path = RESULTS / subfolder / "metricas.csv"
    if not csv_path.exists():
        print(f"  [AVISO] CSV não encontrado para {label} ({csv_path}) — pulando.")
        return None
    df = pd.read_csv(csv_path).rename(columns=COLUMN_ALIASES)
    print(f"  [OK] {label:16} <- resultados/{subfolder}/metricas.csv  ({len(df)} episódios)")
    return df


def moving_avg(series: np.ndarray, window: int = WINDOW):
    """Média móvel; retorna (x_episodios, valores_suavizados)."""
    series = np.asarray(series, dtype=float)
    if len(series) < window:
        return np.arange(1, len(series) + 1), series
    ma = np.convolve(series, np.ones(window) / window, mode="valid")
    x = np.arange(window, len(series) + 1)
    return x, ma


def plot_panel(data: dict, filename: str = "comparativo_painel.png"):
    """Painel 2x2 (recompensa, entregas, taxa de sucesso, colisões) num único slide."""
    specs = [
        ("recompensa", "Recompensa", None),
        ("entregas", "Entregas (de 8)", 8),
        ("taxa_sucesso", "Taxa de sucesso", 0.95),
        ("colisoes", "Colisões", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (col, ylabel, hline) in zip(axes.flat, specs):
        for label, (df, color) in data.items():
            if col in df.columns:
                x, ma = moving_avg(df[col].to_numpy())
                ax.plot(x, ma, color=color, linewidth=1.8, label=label)
        if hline is not None:
            ax.axhline(hline, color="gray", linestyle="--", linewidth=1)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_xlabel("Episódio"); ax.grid(True, alpha=0.3)
    axes.flat[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Comparativo dos métodos — média móvel de 100 episódios",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.name}")


def plot_bars(data: dict, filename: str = "comparativo_barras_final.png"):
    """Barras dos resultados FINAIS (média dos últimos 100 ep.): entregas e recompensa."""
    labels = list(data.keys())
    colors = [c for _, c in data.values()]
    entregas = [data[l][0].tail(WINDOW)["entregas"].mean() for l in labels]
    recompensa = [data[l][0].tail(WINDOW)["recompensa"].mean() for l in labels]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6))
    b1 = a1.bar(labels, entregas, color=colors)
    a1.axhline(8, color="gray", linestyle="--", linewidth=1, label="Meta (8)")
    a1.set_title("Entregas médias (últimos 100 ep.)", fontsize=12, fontweight="bold")
    a1.set_ylabel("Entregas / 8"); a1.set_ylim(0, 8.6); a1.legend()
    a1.bar_label(b1, fmt="%.2f", fontsize=9)

    b2 = a2.bar(labels, recompensa, color=colors)
    a2.axhline(0, color="black", linewidth=0.8)
    a2.set_title("Recompensa média (últimos 100 ep.)", fontsize=12, fontweight="bold")
    a2.set_ylabel("Recompensa")
    a2.bar_label(b2, fmt="%.0f", fontsize=9)

    for ax in (a1, a2):
        ax.tick_params(axis="x", rotation=20); ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Resultados finais por método", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.name}")


def plot_convergence(data: dict, alvo: float = 7.0,
                     filename: str = "comparativo_velocidade.png"):
    """Episódios até a média móvel de entregas atingir `alvo` (velocidade de aprendizado)."""
    labels, valores, colors = [], [], []
    for label, (df, color) in data.items():
        x, ma = moving_avg(df["entregas"].to_numpy())
        atingiu = np.where(ma >= alvo)[0]
        ep = int(x[atingiu[0]]) if len(atingiu) else None  # None = nunca atingiu
        labels.append(label)
        valores.append(ep)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(11, 6))
    plot_v = [v if v is not None else 0 for v in valores]
    bars = ax.bar(labels, plot_v, color=colors)
    for bar, v in zip(bars, valores):
        txt = f"{v} ep" if v is not None else "nunca"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                txt, ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel(f"Episódios até atingir {alvo:.0f}/8 entregas (média móvel)")
    ax.set_title("Velocidade de aprendizado (menor = aprende mais rápido)",
                 fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.name}")


def plot_metric(data: dict, column: str, ylabel: str, title: str,
                filename: str, note: str | None = None):
    """Gera um gráfico comparativo (média móvel) de uma métrica entre métodos."""
    fig, ax = plt.subplots(figsize=(12, 7))
    plotted = False
    for label, (df, color) in data.items():
        if column not in df.columns:
            continue
        x, ma = moving_avg(df[column].to_numpy())
        ax.plot(x, ma, color=color, linewidth=2, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"  [AVISO] nenhum método tem a coluna '{column}' — gráfico {filename} não gerado.")
        return
    ax.set_xlabel("Episódio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    if note:
        ax.text(0.5, -0.12, note, transform=ax.transAxes, ha="center",
                va="top", fontsize=9, style="italic", color="dimgray")
    fig.tight_layout()
    out = OUT_DIR / filename
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {out.name}")


def build_summary(data: dict) -> pd.DataFrame:
    """Tabela-resumo: média dos últimos 100 episódios por método."""
    rows = []
    for label, (df, _color) in data.items():
        tail = df.tail(WINDOW)
        rows.append({
            "metodo": label,
            "episodios": len(df),
            "recompensa_media": round(tail["recompensa"].mean(), 2),
            "entregas_media_de_8": round(tail["entregas"].mean(), 2),
            "taxa_sucesso_frac_caixas": round(tail["taxa_sucesso"].mean(), 3),
            "colisoes_media": round(tail["colisoes"].mean(), 1),
            "melhor_recompensa": round(df["recompensa"].max(), 2),
            "max_entregas": int(df["entregas"].max()),
        })
    return pd.DataFrame(rows)


def main() -> int:
    print("=" * 72)
    print("COMPARATIVO REAL DOS MÉTODOS DE MARL — armazém 12x8 (INF2072)")
    print("=" * 72)
    print(f"Lendo CSVs em: {RESULTS}")
    OUT_DIR.mkdir(exist_ok=True)

    data: dict[str, tuple[pd.DataFrame, str]] = {}
    for label, subfolder, color in METHODS:
        df = load_method(label, subfolder)
        if df is not None:
            data[label] = (df, color)

    if not data:
        print("ERRO: nenhum CSV encontrado. Verifique a pasta resultados/.")
        return 1

    print("\nGerando gráficos comparativos...")
    plot_metric(data, "recompensa", "Recompensa total",
                "Recompensa por Episódio — Comparativo (média móvel 100)",
                "comparativo_recompensa.png")
    plot_metric(data, "entregas", "Entregas (de 8)",
                "Entregas por Episódio — Comparativo (média móvel 100)",
                "comparativo_entregas.png",
                note="Meta = 8 caixas. taxa_sucesso = entregas/8.")
    plot_metric(data, "taxa_sucesso", "Taxa de sucesso (fração das 8 caixas)",
                "Taxa de Sucesso por Episódio — Comparativo (média móvel 100)",
                "comparativo_taxa_sucesso.png")
    plot_metric(data, "colisoes", "Colisões",
                "Colisões por Episódio — Comparativo (média móvel 100)",
                "comparativo_colisoes.png",
                note="Atenção: VDN rodou com MAX_STEPS=1000; os demais com 500 "
                     "(colisões absolutas não são diretamente comparáveis).")
    plot_panel(data)
    plot_bars(data)
    plot_convergence(data)

    print("\nGerando tabela-resumo...")
    summary = build_summary(data)
    out_csv = OUT_DIR / "comparativo_metricas.csv"
    summary.to_csv(out_csv, index=False)
    print(f"  [CSV] {out_csv.name}")
    print()
    print(summary.to_string(index=False))
    print("\nConcluído. Use estes gráficos/tabela na apresentação — NÃO os de _SIMULADO_NAO_USAR/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
