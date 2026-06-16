# LocalMultiAgente — Warehouse MARL

Projeto de **Aprendizado por Reforço Multi-Agente (MARL)** num ambiente customizado de
"armazém": dois robôs numa grade 12×8 precisam pegar caixas (`A`) e entregá-las nos alvos
(`B`), evitando paredes (`X`) e barreiras (`Y`).

O código original estava espalhado e duplicado em vários scripts e notebooks (em
`Código/`). O diretório [`src/`](src) reúne uma versão **modularizada e reutilizável** do
núcleo, contendo o ambiente e dois algoritmos: **IDQN** (Independent Deep Q-Network) e um
**baseline aleatório**.

## Estrutura

```
src/
├── config.py            # MAP_CONFIG, BaseConfig, IDQNConfig, RandomConfig
├── environment.py       # WarehouseEnv (Gymnasium)
├── replay_buffer.py     # PrioritizedReplayBuffer
├── networks.py          # ImprovedDQN (MLP)
├── agents/
│   ├── idqn.py          # IDQNAgent (Double-DQN + PER + soft update)
│   └── random_agent.py  # RandomAgent (baseline)
├── training.py          # loop de treino multi-sessão (reutilizável)
├── evaluation.py        # gravação de vídeo + gráficos consolidados
└── main.py              # interface de linha de comando
```

> Os scripts/notebooks antigos em `Código/` foram mantidos intactos como referência.
> Esta versão modular usa o **ambiente simples** (sem falhas de atuação nem barreiras
> dinâmicas — esses recursos existem apenas nos experimentos em notebook).

## Instalação

```bash
pip install -r requirements.txt
```

Testado com Python 3.13 (ver `.tool-versions`). PyTorch usa GPU (CUDA) automaticamente
quando disponível, caindo para CPU caso contrário.

## Uso

```bash
# Treino rápido de teste (5 episódios)
python -m src.main --algo idqn --episodes 5
python -m src.main --algo random --episodes 5

# Treino completo (1500 episódios por sessão, padrão)
python -m src.main --algo idqn

# Múltiplas sessões com retomada de checkpoint
python -m src.main --algo idqn --sessions 2

# Sem gerar vídeo ao final
python -m src.main --algo random --episodes 5 --no-video
```

Argumentos: `--algo {idqn,random}`, `--sessions N`, `--episodes N` (sobrescreve
`EPISODES_PER_SESSION`), `--output DIR` (sobrescreve `BASE_DIR`), `--no-video`.

## Saídas

Cada execução cria um diretório de resultados (`resultados_warehouse_idqn/` ou
`resultados_warehouse_random/`) com:

- `session_XXXX/metrics/training_metrics.csv` e checkpoints por sessão;
- `consolidated_results/consolidated_metrics.csv` + `consolidated_results.png`;
- `final_results/robot_movement.mp4` (vídeo da política final).

> Esses artefatos são ignorados pelo `.gitignore`. Binários antigos (`.pth`, `.png`,
> `.mp4`) que já estão no histórico do git não são removidos automaticamente pelo
> `.gitignore` — limpá-los do histórico é um passo opcional separado.
