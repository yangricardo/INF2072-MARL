# LocalMultiAgente — Warehouse MARL

Projeto de **Aprendizado por Reforço Multi-Agente (MARL)** num ambiente customizado de
"armazém": dois robôs numa grade 12×8 precisam pegar caixas (`A`) e entregá-las nos alvos
(`B`), evitando paredes (`X`) e barreiras (`Y`).

O código original estava espalhado e duplicado em vários scripts e notebooks (em
`Código/`). O diretório [`src/`](src) reúne uma versão **modularizada e reutilizável** do
núcleo, contendo o ambiente e **seis algoritmos**: **IDQN** (Independent Deep Q-Network),
um **baseline aleatório**, **VDN**, **QMIX**, **MAPPO** e **HATRPO**. Cada módulo cita no
cabeçalho o arquivo de origem de onde foi extraído.

## Estrutura

```
src/
├── config.py            # MAP_CONFIG, BaseConfig e *Config de cada algoritmo
├── environment.py       # WarehouseEnv (Gymnasium) + obs por-robô e estado global
├── replay_buffer.py     # PrioritizedReplayBuffer, VDN e QMIX (variantes)
├── networks.py          # ImprovedDQN, AgentNet, QMixer, Actor/Critic (MAPPO/HATRPO)
├── agents/
│   ├── idqn.py          # IDQNAgent (Double-DQN + PER + soft update)
│   ├── random_agent.py  # RandomAgent (baseline)
│   ├── vdn.py           # VDNController (Q_total = Q1 + Q2) + runner
│   ├── qmix.py          # QMIXAgent + QMixer + QMIXTrainer + runner
│   ├── mappo.py         # MAPPOAgent (PPO, crítico centralizado) + runner
│   └── hatrpo.py        # HATRPOAgent + crítico centralizado + runner
├── training.py          # loop value-based multi-sessão (IDQN/Random)
├── evaluation.py        # gravação de vídeo + gráficos consolidados
└── main.py              # interface de linha de comando (dispatch por --algo)
```

> Os scripts/notebooks antigos em `Código/` foram mantidos intactos como referência.
> Esta versão modular usa o **ambiente simples** (sem falhas de atuação nem barreiras
> dinâmicas — esses recursos existem apenas nos experimentos em notebook), estendido com
> observação por-robô (`_get_observation_for_robot`) e estado global (`_get_global_state`)
> para os algoritmos de crítico centralizado (QMIX/MAPPO/HATRPO).

## Instalação

```bash
pip install -r requirements.txt
```

Testado com Python 3.13 (ver `.tool-versions`). PyTorch usa GPU (CUDA) automaticamente
quando disponível, caindo para CPU caso contrário.

## Uso

```bash
# Treino rápido de teste (3 episódios) — qualquer algoritmo
python -m src.main --algo idqn   --episodes 3
python -m src.main --algo vdn    --episodes 3
python -m src.main --algo qmix   --episodes 3
python -m src.main --algo mappo  --episodes 3
python -m src.main --algo hatrpo --episodes 3
python -m src.main --algo random --episodes 3

# Treino completo (1500 episódios por sessão, padrão)
python -m src.main --algo qmix

# Múltiplas sessões / sem vídeo
python -m src.main --algo idqn --sessions 2
python -m src.main --algo mappo --episodes 3 --no-video
```

Argumentos: `--algo {idqn,random,vdn,qmix,mappo,hatrpo}`, `--sessions N`,
`--episodes N` (sobrescreve `EPISODES_PER_SESSION`), `--output DIR` (sobrescreve
`BASE_DIR`), `--no-video`.

## Saídas

Cada execução cria um diretório de resultados por algoritmo
(`resultados_warehouse_<algo>/`, ex.: `resultados_warehouse_qmix/`) com:

- `session_XXXX/metrics/training_metrics.csv` e checkpoints por sessão;
- `consolidated_results/consolidated_metrics.csv` + `consolidated_results.png`;
- `final_results/robot_movement.mp4` (vídeo da política final).

> Esses artefatos são ignorados pelo `.gitignore`. Binários antigos (`.pth`, `.png`,
> `.mp4`) que já estão no histórico do git não são removidos automaticamente pelo
> `.gitignore` — limpá-los do histórico é um passo opcional separado.
