# LocalMultiAgente — MARL em armazém 12×8 (INF2072 / PUC-DI)

Comparação de métodos de **Aprendizado por Reforço Multiagente** num armazém
logístico simulado: 2 robôs precisam coletar e entregar 8 caixas, com paredes,
barreiras que mudam e falhas estocásticas de atuação.

📄 **Comece por:** [Análise-Reversa-INF2072.md](Análise-Reversa-INF2072.md) — explica
o ambiente, cada algoritmo, os resultados reais e por que cada um deu certo/errado.

---

## Estrutura do repositório

| Pasta | O que é |
|-------|---------|
| [`experimentos/`](experimentos/) | Os 5 notebooks de treino (1 por modelo: IDQN, VDN, MPPO, HATRPO, Randomico). Rodados no Colab+GPU. |
| [`resultados/`](resultados/) | Resultados **reais** de cada modelo. Uma subpasta por modelo. |
| [`comparativos/`](comparativos/) | Gráficos comparativos honestos + tabela-resumo: curvas por métrica, **painel 2×2**, **barras finais** e **velocidade de aprendizado**. |
| [`codigo_analise/`](codigo_analise/) | Scripts de análise (rodam local): comparação, geração de gráficos, **vídeos** e o MAPPO corrigido. |
| `Output/videos/` | **GIFs** das políticas treinadas agindo no armazém (IDQN e MAPPO entregando 8/8; Random caótico). |
| [`apresentacao/`](apresentacao/) | A apresentação (PDF). |
| [`legado/`](legado/) | Versões antigas do código (1.1–1.3, 7.0.0 com lib `rware`, utilitários). Não usado nos resultados finais. |
| [`_SIMULADO_NAO_USAR/`](_SIMULADO_NAO_USAR/) | ⚠️ Notebooks que geravam gráficos **com dados fabricados** (não leem os CSVs reais). **Não usar.** |
| `Modelos/` | `aprendizado_mas_final.pkl` (modelo salvo). |
| `Output/` | Vídeo da simulação + frames da animação. |

### `resultados/<modelo>/` — formato padronizado
Cada subpasta (`random`, `idqn`, `vdn`, `mappo_original`, `mappo_corrigido`) tem:
- `metricas.csv` — métricas dos 3000 episódios.
- `relatorio.txt` — resumo final do treino.
- `graficos/` — gráficos padronizados (série bruta + **média móvel de 100**).
- `graficos_originais/` — PNGs originais do colega (preservados; estilos/nomes variados).
- `models/` — pesos salvos (quando existem).

---

## Resultados (médias dos últimos 100 episódios)

| Modelo | Recompensa | Entregas /8 | Taxa sucesso | Colisões |
|--------|-----------:|------------:|-------------:|---------:|
| **IDQN** ⭐ | 316,05 | 8,00 | 100% | 9,2 |
| **MAPPO (corrigido)** | 313,70 | 8,00 | 100% | 24,8 |
| Random | 85,03 | 3,19 | 39,9% | 138,1 |
| VDN (ajustado) | 44,92 | 1,77 | 22,1% | 261,4 |
| VDN | 7,03 | 1,04 | 13,0% | 807,3 |
| MAPPO (original, c/ bug) | −9,52 | 0,07 | 0,9% | 263,7 |

> Tabela completa: [`comparativos/comparativo_metricas.csv`](comparativos/comparativo_metricas.csv).

**Dois pontos críticos** (detalhados na análise):
1. O **MAPPO original falhou por um bug** de seleção de ação (ε-greedy+argmax numa
   política PPO). Corrigido, ele **empata com o IDQN**.
2. Os comparativos antigos em `_SIMULADO_NAO_USAR/` são **dados fabricados** — use
   só os de `comparativos/` e `resultados/`.

---

## Como reproduzir (local, com a `.venv` do projeto)

```bash
# 1) Gráficos por-modelo padronizados (lê resultados/<modelo>/metricas.csv)
python codigo_analise/gerar_graficos.py

# 2) Gráficos comparativos + painel + barras + velocidade (saída em comparativos/)
python codigo_analise/comparar_metricas.py

# 3) Vídeos (GIF) das políticas treinadas (saída em Output/videos/)
python codigo_analise/gerar_videos.py

# 4) (opcional) re-treinar o MAPPO corrigido — ~5 min na CPU
python codigo_analise/mappo_corrigido.py --episodes 3000 --device cpu --ppo-epochs 8 --minibatch 128
```

> Detalhes dos scripts: [`codigo_analise/README.md`](codigo_analise/README.md).
