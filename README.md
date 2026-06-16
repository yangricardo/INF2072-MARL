# Warehouse MARL

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

---

## Ambiente: Grade e Estrutura de Recompensas

### Grade 12×8

O ambiente `WarehouseEnv` (Gymnasium) simula um armazém representado por uma grade de 12 colunas × 8 linhas. Tipos de célula:

| Símbolo    | Significado                                         |
| ---------- | --------------------------------------------------- |
| `0`        | Espaço livre (transitável)                          |
| `X`        | Parede permanente                                   |
| `Y`        | Barreira estática                                   |
| `R1`, `R2` | Posições iniciais dos robôs                         |
| `A`        | Localização de caixas (origem) — 4 caixas no início |
| `B`        | Alvo de entrega (destino) — 4 alvos fixos           |

Dois robôs partem de cantos opostos no topo da grade. As 4 caixas estão concentradas numa área central e os 4 alvos numa área diferente; o objetivo é que os robôs peguem todas as caixas e as entreguem nos alvos.

### Espaço de Ações

Cada robô possui 6 ações discretas:

| Código | Ação                 |
| ------ | -------------------- |
| 0      | Mover para cima      |
| 1      | Mover para baixo     |
| 2      | Mover para esquerda  |
| 3      | Mover para direita   |
| 4      | Pegar caixa (pickup) |
| 5      | Soltar caixa (drop)  |

### Espaço de Observações

**Observação global (24-dim, `float32`)** — compartilhada entre os agentes:

- Posições normalizadas dos 2 robôs: 2 × 2 = 4 valores
- Posições normalizadas das 4 caixas: 4 × 2 = 8 valores
- Posições normalizadas dos 4 alvos: 4 × 2 = 8 valores
- Distância mínima de cada robô à caixa mais próxima + ao alvo mais próximo: 2 × 2 = 4 valores

**Observação por robô (27-dim)** — usada por IDQN no treino e por MAPPO/HATRPO: observação global
(24-dim) + one-hot de 2 bits identificando o robô (`[1,0]`/`[0,1]`) + 1 flag de inventário
(`1.0` se está carregando uma caixa, `0.0` caso contrário). As dimensões são derivadas do mapa em
runtime (`len(env._get_observation_for_robot(0))`), então mudam automaticamente se o mapa mudar.

Todos os valores são normalizados pela largura/altura da grade.

### Estrutura de Recompensas

| Evento                                         | Recompensa                      |
| ---------------------------------------------- | ------------------------------- |
| Movimento válido                               | −0,005                          |
| Movimento inválido (parede, barreira, colisão) | −0,05                           |
| Pegar caixa (bem-sucedido)                     | +2,0                            |
| Pegar caixa (sem caixa na posição)             | −0,02                           |
| Soltar caixa em alvo —**entrega!**             | +25,0                           |
| Soltar caixa fora de alvo                      | −0,1                            |
| Soltar sem carregar caixa                      | −0,02                           |
| Shaping de proximidade ao objetivo atual       | 0,1 × (d_t − γ·d_{t+1})         |
| **Bônus terminal**: todas as caixas entregues  | +50,0 (dividido entre os robôs) |

O objetivo de cada robô é dinâmico: a caixa mais próxima se está vazio, ou o alvo mais próximo se
está carregando. O _shaping_ é **potential-based** (Ng et al. 1999) com potencial Φ(s) = −d (distância
ao objetivo), aplicando F = γ·Φ(s′) − Φ(s) = d_t − γ·d_{t+1}. Por telescopar ao longo do episódio, não
altera a política ótima e — por ser simétrico — não cria o ciclo de _farming_ do shaping assimétrico
anterior (que rendia ganho líquido ao oscilar perto de uma caixa).

---

## Arquiteturas de Redes Neurais

Todas as redes estão em [`src/networks.py`](src/networks.py).

### ImprovedDQN — usada por IDQN e QMIX

MLP de 3 camadas ocultas com ReLU e Dropout (p=0,2) após cada camada. Inicialização Xavier. Entrada: `state_dim` (22). Saída: `action_dim` (6 Q-values).

### AgentNet — usada por VDN

Arquitetura mais profunda com `LayerNorm` após cada camada oculta e inicialização ortogonal. A última camada é inicializada com ganho 0,01, fazendo os Q-values partirem de zero. Estrutura: `hidden → LayerNorm → ReLU → Dropout → hidden → LayerNorm → ReLU → Dropout → hidden/2 → ReLU → output`.

### QMixer — usada por QMIX

Rede de mistura monotônica condicionada no estado global via **hiper-redes**. Quatro hiper-redes (`hyper_w1`, `hyper_w2`, `hyper_b1`, `hyper_b2`) geram os pesos e vieses da rede de mistura a partir do estado global. A monotonicidade é garantida usando `torch.abs()` nos pesos gerados (W1, W2 ≥ 0), assegurando que ∂Q_total/∂Qi ≥ 0 para todo i.

### ActorNetwork — usada por MAPPO

MLP de 2 camadas ocultas (256 unidades, ReLU). Saída: `softmax` sobre as ações. Retorna `(probs, logits)`.

### CriticNetwork — usada por MAPPO

MLP de 2 camadas (256 unidades). Entrada: estado global (22-dim). Saída: valor escalar V(s).

### ImprovedActorNetwork — usada por HATRPO

Rede com **conexões residuais** (`x = x + camada(x)`), `LayerNorm` e `Dropout` após cada bloco, inicialização ortogonal com ganho 0,01 na camada de saída. Número de camadas configurável (`NUM_LAYERS=3`, `HIDDEN_DIM=512`). Saída: `(probs, logits)`.

### ImprovedCriticNetwork — usada por HATRPO

Mesma arquitetura residual da rede de ator. Acompanhada de uma **rede-alvo** (`critic_target`) atualizada via soft update com τ=0,005, estabilizando o treinamento.

---

## Algoritmos

### Baseline Aleatório (Random)

> Implementação: [src/agents/random_agent.py](src/agents/random_agent.py) ([`select_action` linha 21](src/agents/random_agent.py#L21))

**O que é:** Um agente sem aprendizado que seleciona ações uniformemente ao acaso. Serve como limite inferior de desempenho — qualquer algoritmo de RL deve superá-lo.

**Como funciona:**

- A cada passo, chama `random.randrange(action_dim)`, sem rede neural, sem buffer.
- A interface é idêntica à do `IDQNAgent` (métodos `remember`/`optimize` são no-ops), permitindo uso no mesmo loop de treino genérico.

**Hiperparâmetros:** Apenas os do ambiente (`MAX_STEPS=500`, `EPISODES_PER_SESSION=1500`). Sem hiperparâmetros de aprendizado.

---

### IDQN — Independent Deep Q-Network

> Implementação: [src/agents/idqn.py](src/agents/idqn.py) · [src/replay_buffer.py](src/replay_buffer.py) (`PrioritizedReplayBuffer`)

**O que é:** Cada robô treina uma Q-network independente sem coordenação explícita. É o ponto de partida mais simples do paradigma MARL: múltiplos agentes DQN treinando em paralelo no mesmo ambiente.

**Como funciona:**

- Cada `IDQNAgent` mantém uma `policy_net` e uma `target_net` (`ImprovedDQN`).
- **Seleção de ação:** epsilon-greedy com decaimento linear de ε: 1,0 → 0,05 em 200 000 passos ([`get_epsilon`](src/agents/idqn.py)).
- **Memória:** `PrioritizedReplayBuffer` (PER) com α=0,6, β: 0,6 → 1,0 (annealing linear) — transições com maior erro TD são amostradas com maior frequência.

**Prioridade normalizada e importance-sampling weights** (Schaul et al. 2015) — [`replay_buffer.py` linhas 37–45](src/replay_buffer.py#L37-L45):

```math
p_i = \frac{|\delta_i|^\alpha}{\sum_j |\delta_j|^\alpha}, \qquad w_i = \frac{(N \cdot p_i)^{-\beta}}{\max_j w_j}
```

- **Atualização (Double DQN):** a `policy_net` seleciona a ação greedy; a `target_net` avalia o valor, reduzindo viés de superestimação.

**TD-target** (Mnih et al. 2015; van Hasselt et al. 2016) — [`idqn.py` linhas 118–123](src/agents/idqn.py#L118-L123):

```math
y_i = r_i + \gamma \, Q_{\theta^-}\!\left(s'_i,\, \arg\max_{a'} Q_\theta(s'_i, a')\right) \cdot (1 - d_i)
```

**Loss com pesos PER** — [`idqn.py` linhas 130–132](src/agents/idqn.py#L130-L132):

```math
\mathcal{L}(\theta) = \mathbb{E}_i\!\left[ w_i \cdot \bigl(y_i - Q_\theta(s_i, a_i)\bigr)^2 \right]
```

**Soft update** (Polyak averaging, τ=0,005) — [`idqn.py`](src/agents/idqn.py):

```math
\theta^{-} \leftarrow \tau\,\theta + (1-\tau)\,\theta^{-}
```

**Hiperparâmetros principais:**

| Parâmetro       | Valor   |
| --------------- | ------- |
| Learning rate   | 0,0001  |
| Batch size      | 64      |
| γ (desconto)    | 0,99    |
| τ (soft update) | 0,005   |
| Hidden dim      | 512     |
| Dropout         | 0,0     |
| Buffer size     | 100 000 |
| ε_decay_steps   | 200 000 |

**Referências:**

- Mnih, V. et al. (2015). _Human-level control through deep reinforcement learning_. Nature. [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)
- van Hasselt, H., Guez, A., & Silver, D. (2016). _Deep Reinforcement Learning with Double Q-learning_. AAAI. [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)
- Schaul, T. et al. (2015). _Prioritized Experience Replay_. ICLR 2016. [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)

---

### VDN — Value Decomposition Networks

**O que é:** Método CTDE (_Centralized Training, Decentralized Execution_) que fatora a função Q conjunta como **soma** das Q-functions individuais: Q_total = Q₁(s, a₁) + Q₂(s, a₂). Isso permite que cada agente execute de forma descentralizada (apenas com sua Q-function) enquanto o treinamento é centralizado.

**Como funciona:**

- `VDNController` centraliza ambas as redes (`policy_nets: ModuleList[AgentNet]`) num único otimizador Adam.
- Transições **conjuntas** `(s, [a₁,a₂], [r₁,r₂], s', done)` são armazenadas no `VDNPrioritizedReplayBuffer`.
- **Fatoração aditiva (VDN):** soma dos Q-values individuais, permitindo execução descentralizada (Sunehag et al. 2017):

```math
Q_{\text{tot}}(s, a_1, a_2) = Q_1(o_1, a_1) + Q_2(o_2, a_2)
```

- **Individual-Global-Max (IGM):** A fatoração aditiva garante decomponibilidade do argmax:

```math
\arg\max_{\mathbf{a}} Q_{\text{tot}} = \bigl(\arg\max_{a_1} Q_1,\; \arg\max_{a_2} Q_2\bigr)
```

  Isto preserva execução descentralizada: cada agente só precisa de sua própria Q-function.

- **Target (Double-DQN por agente):** a `policy_net` seleciona a ação greedy; a `target_net` avalia:

```math
y = \sum_{i=1}^{2} r_i + \gamma \sum_{i=1}^{2} \max_{a'_i} Q_i^{-}(o'_i, a'_i) \cdot (1 - d)
```
  
- Learning rate com _cosine annealing_ ao longo de todos os episódios. Soft update das target nets com τ=0,005.

**Hiperparâmetros principais:**

| Parâmetro       | Valor   |
| --------------- | ------- |
| Learning rate   | 0,0003  |
| Batch size      | 256     |
| γ (desconto)    | 0,97    |
| τ (soft update) | 0,005   |
| Hidden dim      | 256     |
| Buffer size     | 200 000 |
| ε_decay_steps   | 120 000 |

**Referência:**

- Sunehag, P. et al. (2017). _Value-Decomposition Networks For Cooperative Multi-Agent Learning_. AAMAS 2018. [https://arxiv.org/abs/1706.05296](https://arxiv.org/abs/1706.05296)

---

### QMIX — Monotonic Value Function Factorisation

> Implementação: [src/agents/qmix.py](src/agents/qmix.py) · [src/networks.py](src/networks.py) (`QMixer`)

**O que é:** Estende o VDN ao permitir pesos de mistura não-lineares mas **monotônicos**: Q_total = f(Q₁, Q₂; estado_global), onde f é uma rede de mistura (_mixer_) condicionada no estado global via hiper-redes. A monotonicidade garante que o argmax sobre Q_total é equivalente ao argmax sobre as Q-functions individuais, preservando a execução descentralizada.

**Como funciona:**

- Cada `QMIXAgent` mantém `policy_net` e `target_net` (ImprovedDQN).
- O `QMixer` recebe Q-values individuais e o estado global; hiper-redes geram pesos W1, W2 ≥ 0 (via `abs()`) garantindo monotonicidade.
- `QMIXPrioritizedReplayBuffer` armazena transições enriquecidas com estados globais: `(s, [a₁,a₂], [r₁,r₂], s', done, global_s, next_global_s)`.

**Rede de mistura** (Rashid et al. 2018) — pesos W₁, w₂ ≥ 0 via `abs()` nas hiper-redes garantem monotonicidade — [`networks.py` linhas 114–126](src/networks.py#L114-L126):

```math
Q_{\text{tot}}(\mathbf{q}, s) = \mathbf{w}_2(s)^\top \mathrm{ReLU}\!\left(\mathbf{W}_1(s)\,\mathbf{q} + \mathbf{b}_1(s)\right) + b_2(s)
```

```math
\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0 \quad \forall\, i
```

**`QMIXTrainer.optimize()`** — loss MSE ponderada por PER — [`qmix.py` linhas 149–163](src/agents/qmix.py#L149-L163):

```math
\mathcal{L}_{\text{mixer}} = \mathbb{E}\!\left[w \cdot \bigl(y - Q_{\text{tot}}^\theta\bigr)^2\right]
```

```math
y = \sum_i r_i + \gamma\, Q_{\text{tot}}^{-}(s', \mathbf{a}^*) \cdot (1-d)
```

Por agente: loss contrafactual ([`qmix.py` linha 169](src/agents/qmix.py#L169)) usando $(y - Q_{\text{tot}})^{\text{detach}} \cdot Q_i$. Soft update das target nets com τ=0,001 ([`qmix.py` linhas 85–92](src/agents/qmix.py#L85-L92)).

**Hiperparâmetros principais:**

| Parâmetro        | Valor   |
| ---------------- | ------- |
| Learning rate    | 0,0001  |
| Batch size       | 128     |
| γ (desconto)     | 0,95    |
| τ (soft update)  | 0,001   |
| Hidden dim       | 256     |
| Mixer hidden dim | 128     |
| Buffer size      | 100 000 |

**Referência:**

- Rashid, T. et al. (2018). _QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning_. ICML 2018. [https://arxiv.org/abs/1803.11605](https://arxiv.org/abs/1803.11605)

---

### MAPPO — Multi-Agent PPO

> Implementação: [src/agents/mappo.py](src/agents/mappo.py)

**O que é:** Extensão multi-agente do PPO (_Proximal Policy Optimization_) no paradigma CTDE. Cada agente possui um **ator individual** (política descentralizada); um **crítico centralizado** compartilha o estado global para estimar V(s) durante o treinamento.

**Como funciona:**

- `MAPPOAgent` contém `ActorNetwork` (entrada: obs local 24-dim) e `CriticNetwork` (entrada: estado global 22-dim), com otimizadores Adam separados.
- **Coleta:** por episódio, armazena estados locais, ações, log-probs, recompensas e estados globais.

**Atualização pós-episódio:**

**1.** Computa V(s) com o crítico centralizado.

**2. GAE** — Generalized Advantage Estimation (Schulman et al. 2016) — [`mappo.py` linhas 86–96](src/agents/mappo.py#L86-L96):

```math
\delta_t = r_t + \gamma\,V(s_{t+1})(1-d_t) - V(s_t)
```

```math
\hat{A}_t = \sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}
```

```math
\hat{R}_t = \hat{A}_t + V(s_t)
```

```math
\hat{A} \leftarrow \frac{\hat{A} - \mu_{\hat{A}}}{\sigma_{\hat{A}} + \varepsilon}
```

**3. PPO_EPOCHS=10** repetições com mini-batches de 32 — [`mappo.py` linhas 140–156](src/agents/mappo.py#L140-L156):

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
```

```math
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[\min\!\left(r_t\,\hat{A}_t,\;\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\,\hat{A}_t\right)\right]
```

```math
\mathcal{L}_{\text{actor}} = -\mathcal{L}^{\text{CLIP}} - c_H\,H[\pi]
```

```math
H[\pi] = -\sum_{a}\pi(a \mid s)\log\pi(a \mid s)
```

```math
\mathcal{L}_{\text{critic}} = \mathbb{E}\!\left[\bigl(V(s) - \hat{R}_t\bigr)^2\right]
```

**4.** Clip de gradiente em 0,5 para ator e crítico. Decaimento multiplicativo de epsilon: $\varepsilon \leftarrow \varepsilon \times 0.995$ por episódio.

**Hiperparâmetros principais:**

| Parâmetro       | Valor |
| --------------- | ----- |
| Actor LR        | 3e-4  |
| Critic LR       | 3e-4  |
| γ (desconto)    | 0,99  |
| λ (GAE)         | 0,95  |
| PPO clip ε      | 0,2   |
| Entropia coef.  | 0,01  |
| PPO epochs      | 10    |
| Mini-batch size | 32    |

**Referência:**

- Yu, C. et al. (2022). _The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games_. NeurIPS 2022. [https://arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)

---

### HATRPO — Hierarchical Actor Trust-Region Policy Optimisation

> Implementação: [src/agents/hatrpo.py](src/agents/hatrpo.py)

**O que é:** Variante multi-agente do TRPO (_Trust Region Policy Optimization_) com restrição de região de confiança por agente. Os atores usam redes residuais e treinam **sequencialmente** (um agente por vez) usando vantagens estimadas por um crítico centralizado compartilhado — essa atualização sequencial é a característica "hierárquica" do algoritmo.

**Como funciona:**

- `HATRPOAgentOptimized`: `ImprovedActorNetwork` (residual) + rede-alvo `actor_old` para monitorar divergência de política.
- `CentralizedCriticOptimized`: `ImprovedCriticNetwork` (residual) com soft update (τ=0,005) da target; responsável pelo GAE.
- `TrajectoryBuffer`: acumula a trajetória completa do episódio com estados achatados de ambos os agentes.

**Atualização pós-episódio:**

**1. Crítico** — MSE loss + Polyak soft update — [`hatrpo.py` linhas 138–157](src/agents/hatrpo.py#L138-L157):

```math
\mathcal{L}_{\text{critic}} = \mathbb{E}\!\left[\bigl(V(s) - \hat{R}_t\bigr)^2\right]
```

```math
\theta_{\text{target}} \leftarrow (1-\tau)\,\theta_{\text{target}} + \tau\,\theta
```

**2. Por agente** (sequencial — HATRPO original de Kuba et al. 2021):

_Formulação teórica_ — restrição de região de confiança via KL-divergence:

```math
\max_{\pi_i}\;\mathbb{E}\!\left[A_i(s,\mathbf{a})\right]
```

```math
\text{s.t.} \quad \mathbb{E}_s\!\left[D_{\mathrm{KL}}\!\left(\pi_i^{\text{old}}(\cdot \mid s)\,\|\,\pi_i(\cdot \mid s)\right)\right] \leq \delta
```

_Implementação prática_ — PPO clip como aproximação ao trust-region (ε=0,2 ≈ MAX_KL=0,02) — [`hatrpo.py` linhas 89–99](src/agents/hatrpo.py#L89-L99):

```math
r_t(\theta_i) = \frac{\pi_{\theta_i}(a \mid s)}{\pi_{\theta_i^{\text{old}}}(a \mid s)}
```

```math
\mathcal{L}_{\text{actor}} = -\mathbb{E}_t\!\left[\min\!\left(r_t\,\hat{A}_t,\;\mathrm{clip}(r_t,1-\varepsilon,1+\varepsilon)\,\hat{A}_t\right)\right] - c_H\,H[\pi]
```

GAE do crítico centralizado — [`hatrpo.py` linhas 161–173](src/agents/hatrpo.py#L161-L173). A cada `TARGET_UPDATE_FREQ=100` passos: `actor_old.load_state_dict(actor.state_dict())`

**Hiperparâmetros principais:**

| Parâmetro      | Valor |
| -------------- | ----- |
| Actor LR       | 3e-4  |
| Critic LR      | 3e-3  |
| γ (desconto)   | 0,99  |
| λ (GAE)        | 0,95  |
| Max KL         | 0,02  |
| Hidden dim     | 512   |
| Num layers     | 3     |
| Entropia coef. | 0,01  |

**Referência:**

- Kuba, J. G. et al. (2021). _Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning_. ICLR 2022. [https://arxiv.org/abs/2109.11251](https://arxiv.org/abs/2109.11251)

---

## Diagramas UML

### Visão Geral da Arquitetura

![Arquitetura Geral](Esquemáticos/architecture_overview.png)

### Diagramas de Classes

**Infraestrutura (config, environment, networks, buffers):**

![Classes — Infraestrutura](Esquemáticos/class_diagram.png)

**Agentes (IDQN, VDN, QMIX, MAPPO, HATRPO):**

![Classes — Agentes](Esquemáticos/class_diagram_agents.png)

### Fluxos de Treino

| Diagrama                                                                | Tipo        | Descrição                                                                                  |
| ----------------------------------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------ |
| [`sequence_qmix.puml`](Esquemáticos/sequence_qmix.puml)                 | Sequência   | Um passo de treino QMIX: epsilon-greedy → step → optimize (mixer + loss contrafactual)     |
| [`sequence_mappo.puml`](Esquemáticos/sequence_mappo.puml)               | Sequência   | Episódio e atualização MAPPO: coleta de trajetória → GAE → PPO multi-época                 |
| [`sequence_hatrpo.puml`](Esquemáticos/sequence_hatrpo.puml)             | Sequência   | Episódio e atualização HATRPO: buffer de trajetória → GAE → trust-region update sequencial |
| [`environment_flow.puml`](Esquemáticos/environment_flow.puml)           | Atividade   | Fluxo de `WarehouseEnv.step()`: movimento, interação, shaping e bônus terminal             |
| [`training_pipeline.puml`](Esquemáticos/training_pipeline.puml)         | Atividade   | Pipeline multi-sessão de `training.py` (IDQN/Random): checkpoints, métricas e vídeo        |

**Para gerar as imagens PNG dos diagramas**, instale o [PlantUML](https://plantuml.com/starting) e execute:

```bash
plantuml Esquemáticos/*.puml
```

Ou visualize online em [plantuml.com/plantuml](https://www.plantuml.com/plantuml/uml/).

---

## Referências

- Mnih, V., Kavukcuoglu, K., Silver, D. et al. (2015). _Human-level control through deep reinforcement learning_. Nature, 518, 529–533. [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)
- van Hasselt, H., Guez, A., & Silver, D. (2016). _Deep Reinforcement Learning with Double Q-learning_. AAAI 2016. [https://arxiv.org/abs/1509.06461](https://arxiv.org/abs/1509.06461)
- Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). _Prioritized Experience Replay_. ICLR 2016. [https://arxiv.org/abs/1511.05952](https://arxiv.org/abs/1511.05952)
- Sunehag, P., Lever, G., Gruslys, A. et al. (2017). _Value-Decomposition Networks For Cooperative Multi-Agent Learning_. AAMAS 2018. [https://arxiv.org/abs/1706.05296](https://arxiv.org/abs/1706.05296)
- Rashid, T., Samvelyan, M., de Witt, C. S. et al. (2018). _QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning_. ICML 2018. [https://arxiv.org/abs/1803.11605](https://arxiv.org/abs/1803.11605)
- Yu, C., Velu, A., Vinitsky, E. et al. (2022). _The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games_. NeurIPS 2022. [https://arxiv.org/abs/2103.01955](https://arxiv.org/abs/2103.01955)
- Kuba, J. G., Chen, R., Wen, M. et al. (2021). _Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning_. ICLR 2022. [https://arxiv.org/abs/2109.11251](https://arxiv.org/abs/2109.11251)
