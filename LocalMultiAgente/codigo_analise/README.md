# codigo_analise — scripts de análise (dados REAIS)

Este diretório substitui os gráficos comparativos fabricados dos notebooks
`_SIMULADO_NAO_USAR/Unifica*.ipynb` por análises fiéis aos CSVs reais.

## Scripts

| Script | O que faz | Saída |
|--------|-----------|-------|
| `comparar_metricas.py` | Curvas comparativas + painel 2×2 + barras finais + velocidade de aprendizado + tabela-resumo | `comparativos/` |
| `gerar_graficos.py` | Gráficos por-modelo padronizados (série bruta + média móvel de 100) | `resultados/<modelo>/graficos/` |
| `gerar_videos.py` | GIFs das políticas treinadas agindo no armazém (inferência nos modelos salvos) | `Output/videos/` |
| `mappo_corrigido.py` | MAPPO com o bug de seleção de ação consertado (empata com IDQN) | `resultados/mappo_corrigido/` |

```bash
python comparar_metricas.py && python gerar_graficos.py && python gerar_videos.py
```

## Por quê?

Os notebooks em `_SIMULADO_NAO_USAR/` **não leem os CSVs de treinamento**. Eles
desenham curvas **simuladas**, usando `np.random.normal(...)` e dicionários de
valores digitados à mão. Essas curvas inventadas chegam a **contradizer os
relatórios reais** — por exemplo, o MAPPO real entrega apenas 0,07/8 caixas, mas
o gráfico simulado mostra a "taxa de sucesso" do MAPPO subindo para ~0,4.

Apresentar esses gráficos seria incorreto e um risco de integridade acadêmica.

## O que `comparar_metricas.py` faz

Lê os CSVs reais gravados por cada experimento (em `resultados/<modelo>/`) e gera, de forma fiel aos dados:

- `comparativo_recompensa.png`
- `comparativo_entregas.png`
- `comparativo_taxa_sucesso.png`
- `comparativo_colisoes.png`
- `comparativo_metricas.csv` (tabela-resumo, média dos últimos 100 episódios)

## Como rodar

Localmente (a `.venv` do projeto já tem pandas + matplotlib):

```bash
cd "LocalMultiAgente/codigo_analise"
python comparar_metricas.py
```

Não depende do Google Colab nem do Drive — usa caminhos relativos.

## `mappo_corrigido.py` — MAPPO com o bug consertado

O MAPPO original (em `experimentos/Executa Experimento MPPO.ipynb`) tinha um **bug na
seleção de ação**: usava ε-greedy + `argmax` numa política PPO (estocástica),
registrando log_probs inválidos → a razão de importância do PPO virava ruído e o
modelo ficava **pior que o aleatório** (−9,5 de recompensa, 0,07/8 entregas).

`mappo_corrigido.py` é uma cópia fiel desse notebook (mesmo ambiente, mesmas
redes) com **apenas** a correção: amostragem categórica correta
(`Categorical(probs).sample()` + log_prob dessa amostragem), sem ε-greedy/argmax.
Também alivia o update (PPO_EPOCHS 20→8, minibatch 16→128) para ganhar
estabilidade e velocidade.

Resultado: o MAPPO passa a **empatar com o IDQN** — recompensa **313,70**,
**8,00/8 entregas, 100% de sucesso** — provando que o fracasso era um bug.

```bash
# treino completo, ~5 min na CPU de um Mac Apple Silicon
python mappo_corrigido.py --episodes 3000 --device cpu --ppo-epochs 8 --minibatch 128
```

Saída em `resultados/mappo_corrigido/` (mesmo formato dos outros experimentos). O
`comparar_metricas.py` já inclui esse resultado como **"MAPPO (corrig.)"** ao lado
do **"MAPPO (orig.)"**, mostrando o antes/depois.

> Atenção: rodar no **MPS** (GPU do Mac) ficou ~300× mais lento por overhead de
> dispatch em tensores pequenos. Use `--device cpu`.

## Notas de honestidade metodológica

- **`taxa_sucesso` = entregas / 8** (fração das 8 caixas entregues no episódio).
  É a mesma definição nos quatro notebooks (`WarehouseEnv._get_info()`).
- **HATRPO ficou de fora**: o algoritmo foi implementado, mas não há execução
  salva (sem pasta de resultados / CSV).
- **VDN rodou com `MAX_STEPS = 1000`**, enquanto os demais usaram `500`. Por isso
  o número absoluto de colisões do VDN não é diretamente comparável — o gráfico
  de colisões traz essa ressalva.
- Cada método também variou hiperparâmetros e até magnitudes de reward
  (ver `Análise-Reversa-INF2072.md`), então o comparativo é **indicativo**, não
  um benchmark perfeitamente controlado. Para um benchmark rigoroso seria preciso
  fixar ambiente, seeds e protocolo idênticos entre todos.
