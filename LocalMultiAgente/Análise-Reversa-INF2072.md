# Análise de Engenharia Reversa — Projeto LocalMultiAgente (INF2072 / PUC-DI)

> Material de estudo para apresentação. Reconstrói, a partir do código, **o que
> foi construído**, **quais resultados foram obtidos** e **o que dá para fazer a
> seguir**. Todas as afirmações técnicas foram conferidas no código-fonte dos
> notebooks em `LocalMultiAgente/experimentos/`.

---

## 1. Visão geral do problema

O projeto estuda **Aprendizado por Reforço Multiagente (MARL)** num cenário de
**armazém logístico simulado**. Dois robôs (`R1`, `R2`) precisam **coletar 8
caixas e entregá-las** na zona de entrega, num mapa em grade 12×8 com paredes,
barreiras que mudam e **falhas estocásticas de atuação** (os robôs nem sempre
obedecem ao comando).

O objetivo do trabalho é **comparar diferentes algoritmos de MARL** no mesmo
ambiente e entender qual aprende a tarefa e por quê.

**Métodos avaliados:** Random (baseline), **IDQN**, **VDN**, **MAPPO** e
**HATRPO** (este último implementado, mas sem execução salva).

---

## 2. O ambiente — `WarehouseEnv`

Ambiente customizado seguindo a interface do **Gymnasium** (`gym.Env`). Cada
notebook traz a sua própria cópia da classe (são *self-contained*); a definição
canônica está em
[`experimentos/Executa Experimento IDQN.ipynb`](experimentos/Executa%20Experimento%20IDQN.ipynb).

### 2.1 O mapa (grade 12×8, `MAP_CONFIG`)

| Símbolo | Significado |
|--------|-------------|
| `R1` / `R2` | posição inicial dos 2 robôs |
| `A` | caixa a ser coletada (são **8** no total) |
| `B` | célula de entrega / alvo (são **8**) |
| `X` | parede fixa (intransponível) |
| `Y` | **barreira removível** — a cada `reset()`, 50% de chance de virar caminho livre |
| `0` | célula livre |

As barreiras `Y` introduzem **variação entre episódios**: o layout muda um pouco
toda vez, forçando políticas mais robustas (`_remove_random_y_barriers`).

### 2.2 Ações (6 por robô)

`0=CIMA  1=BAIXO  2=ESQUERDA  3=DIREITA  4=PEGAR  5=SOLTAR`.
O espaço de ação é um `Tuple` de dois `Discrete(6)` (uma ação por robô).

### 2.3 Falhas estocásticas (`_move_robot_with_failure`)

Com probabilidade `FAILURE_PROBABILITY` (15–20% conforme o experimento) o
movimento **falha**: o robô recebe penalidade e, se `ALTERNATIVE_DIRECTIONS`
estiver ligado, é **desviado para uma direção alternativa válida**. Isso modela
incerteza de atuação (um robô real que escorrega/derrapa) e é o que torna a
tarefa genuinamente difícil.

### 2.4 Função de recompensa (a peça mais importante)

A recompensa é **densa** ("reward shaping"): além dos eventos de sucesso, o
agente é guiado a cada passo pela distância às caixas.

| Evento | Recompensa | Onde no código |
|--------|-----------:|----------------|
| Pegar uma caixa (`PEGAR` válido) | **+2,0** | `_pickup_box` |
| Entregar uma caixa no alvo (`SOLTAR` válido) | **+25,0** | `_drop_box` |
| Entregar **todas** as 8 caixas | **+50,0** (bônus) | `_calculate_shaped_reward` |
| Aproximar-se da caixa mais próxima | **+0,1 × Δdist** | `_calculate_shaped_reward` |
| Afastar-se da caixa | −0,02 × Δdist | `_calculate_shaped_reward` |
| Passo válido (andar) | −0,005 | `_move_robot_with_failure` |
| Bater em parede/limite (colisão) | −0,02 | `_move_robot_with_failure` |
| Falha de atuação | −0,15 (ou −0,10) | `FAILURE_PENALTY` |
| `PEGAR`/`SOLTAR` inválido | −0,02 / −0,05 | `_pickup_box` / `_drop_box` |

> **Por que isso importa para a apresentação:** o shaping baseado em distância é
> **forte**. Ele praticamente "puxa" o robô até as caixas. Isso explica por que
> até o agente **aleatório** consegue ~3 entregas e por que o método mais simples
> (IDQN) aprende tão bem — voltaremos a isso na Seção 5.

### 2.5 Observação (`_get_observation`)

Vetor de floats **normalizados** contendo: posição dos 2 robôs, posição de cada
caixa (ou `-1,-1` se já entregue), posição de cada alvo, e as distâncias mínimas
de cada robô à caixa e ao alvo mais próximos. A observação é **global** (cada
agente "enxerga" o estado todo).

### 2.6 Término

- `terminated`: todas as 8 caixas entregues.
- `truncated`: atingiu `MAX_STEPS` (500 na maioria; **1000 no VDN**).

---

## 3. Os métodos comparados

Todos compartilham o mesmo ambiente. A diferença está em **como cada robô decide
a ação** e em **como o aprendizado é coordenado** (descentralizado vs.
*centralized training, decentralized execution* — CTDE).

### 3.1 Random (baseline)
- **Ideia:** ações totalmente aleatórias. Serve de **piso de comparação**.
- **Código:** `RandomAgent` em
  [`Executa Experimento Randomico.ipynb`](experimentos/Executa%20Experimento%20Randomico.ipynb).
- Treina "instantaneamente" (0,5 min) — não há aprendizado, só medição.

### 3.2 IDQN — Independent Deep Q-Network ⭐
- **Ideia:** cada robô tem **sua própria rede DQN** e aprende de forma
  **independente**, tratando o outro robô como parte do ambiente.
- **Extras implementados** (vão além de um DQN básico — bons pontos a citar):
  - **Double DQN** (seleção de ação pela `policy_net`, avaliação pela `target_net`).
  - **Prioritized Experience Replay** (`PrioritizedReplayBuffer`, α=0.6, β anelado).
  - **Soft update** da target network (`TAU`), dropout e clipping de gradiente.
- **Código:** `IDQNAgent`, `DQN`, `PrioritizedReplayBuffer`.
- **Hiperparâmetros:** LR=1e-4, γ=0.95, hidden=512, batch=256, ε:1.0→0.05 em 100k
  passos, falhas 20%.

### 3.3 VDN — Value Decomposition Networks
- **Ideia (CTDE):** treina de forma centralizada decompondo o valor da equipe de
  forma **aditiva**: `Q_total = Q1(o,a1) + Q2(o,a2)`. As redes individuais são
  atualizadas por um **único loss/otimizador** sobre `Q_total`, mas cada robô
  **age de forma descentralizada** (argmax do seu próprio Q).
- **Código:** `VDNController`, `AgentNet`.
- **Hiperparâmetros:** LR=3e-4, γ=0.97, hidden=256, **MAX_STEPS=1000**,
  TRAIN_FREQ=2, falhas 15%.

### 3.4 MAPPO — Multi-Agent PPO
- **Ideia (CTDE):** método **ator-crítico on-policy**. Cada robô tem uma política
  (ator); um **crítico** estima o valor. Usa **PPO** (clipping da razão de
  probabilidades, GAE) para atualizações estáveis.
- **Código:** `ActorNetwork`, `CriticNetwork`, `MAPPOAgent`.
- **Hiperparâmetros:** actor/critic LR=1e-4, γ=0.99, λ=0.95, clip=0.2,
  entropy=0.05, **PPO_EPOCHS=20**, batch=32.

### 3.5 HATRPO — Heterogeneous-Agent TRPO
- **Ideia (CTDE):** atualização de política com **região de confiança** (KL
  limitado), crítico centralizado e atualização sequencial entre agentes.
- **Código:** `HATRPOAgentOptimized`, `CentralizedCriticOptimized`,
  `TrajectoryBuffer`. **Reescalou as recompensas** (pickup=10, drop=50, bônus=100).
- **Status:** ⚠️ **implementado, mas sem execução salva** (não há pasta de
  resultados). Não entra na comparação quantitativa.

---

## 4. Resultados reais (médias dos últimos 100 episódios)

Valores extraídos diretamente dos CSVs de treinamento e conferidos com os
`relatorio_final*.txt`. Reproduzidos por
[`codigo_analise/comparar_metricas.py`](codigo_analise/comparar_metricas.py).

| Método | Recompensa | Entregas /8 | Taxa sucesso* | Colisões | Tempo treino |
|--------|-----------:|------------:|--------------:|---------:|-------------:|
| **IDQN** ⭐ | **316,05** | **8,00** | **100%** | **9,2** | 217 min |
| **MAPPO (corrigido)** ‡ | **313,70** | **8,00** | **100%** | **24,8** | 5 min |
| Random | 85,03 | 3,19 | 39,9% | 138,1 | 0,5 min |
| VDN (ajustado) § | 44,92 | 1,77 | 22,1% | 261,4 | 651 min |
| VDN | 7,03 | 1,04 | 13,0% | 807,3 † | 326 min |
| MAPPO (original, com bug) | −9,52 | 0,07 | 0,9% | 263,7 | 306 min |

‡ **MAPPO corrigido** = mesmo código, com o bug de seleção de ação consertado
(ver Seção 5). Roda local em ~5 min via `codigo_analise/mappo_corrigido.py`.
§ **VDN ajustado** = VDN com `MAX_STEPS=500` + ID do agente na observação
(ver Seção 5). Melhorou bastante (colisões 807→261), mas **continua fracassando**.

\* `taxa_sucesso = entregas/8` (fração das 8 caixas entregues no episódio) — mesma
definição nos experimentos.
† VDN (original) rodou com `MAX_STEPS=1000` (os outros, 500); colisões absolutas
não são diretamente comparáveis.

Gráficos comparativos honestos (média móvel de 100 episódios) em
`comparativos/comparativo_*.png`.

---

## 5. Análise — por que esses resultados? (o ponto alto da banca)

### Por que o IDQN venceu (e por margem enorme)
1. **O reward shaping faz o trabalho pesado.** Como há recompensa densa por
   aproximação + entrega, cada robô tem um sinal claro a cada passo. Não é preciso
   coordenação sofisticada — basta cada um "seguir o gradiente" até as caixas.
2. **Independência ≈ dois sub-problemas mais simples.** Com 2 robôs e tarefa
   amplamente paralelizável (há caixas de sobra), tratar o outro como ambiente
   funciona bem; o problema de cada agente fica quase-estacionário.
3. **A "caixa de ferramentas" do IDQN é robusta:** Double DQN reduz
   superestimação, PER foca nas transições informativas, soft update estabiliza
   o alvo. São justamente os truques que mais ajudam em recompensa densa.

### Por que VDN e MAPPO falharam
- **Não-estacionariedade + crédito mal atribuído.** Métodos CTDE precisam separar
  "quem causou a recompensa da equipe". Com `Q_total = Q1+Q2` (VDN) ou um crítico
  compartilhado (MAPPO), o sinal de cada robô fica ruidoso enquanto os dois mudam
  de política ao mesmo tempo.
- **VDN — política degenerada.** 807 colisões/episódio (com 1000 passos) indica
  robôs presos batendo em parede: a decomposição aditiva não conseguiu produzir
  Q's individuais úteis. A recompensa máxima de 373 mostra que *às vezes* acertou
  — ou seja, **instabilidade**, não incapacidade.

#### 🔬 VDN ajustado — testamos as duas hipóteses (e o VDN ainda falha)
Levantamos duas suspeitas sobre o VDN: (a) a **observação simétrica** (as duas
redes recebiam input idêntico → políticas iguais → disputam a mesma caixa) e (b)
o **`MAX_STEPS=1000`** desalinhado. Corrigimos as duas: adicionamos um **one-hot
do ID do agente** na observação (quebra a simetria) e alinhamos **`MAX_STEPS=500`**.
Resultado (`codigo_analise/vdn_ajustado.py`, treino completo de 3000 ep):

| | VDN original | VDN ajustado |
|---|---:|---:|
| Recompensa | 7,03 | **44,92** |
| Entregas | 1,04/8 | **1,77/8** |
| Colisões | 807 | **261** |
| Sucesso completo | 0% | **0%** |

Ou seja: os ajustes **ajudaram de verdade** (colisões caíram 3×, recompensa e
entregas subiram), mas o VDN **continua sem resolver a tarefa** (1,77/8, 0% de
sucesso completo). **Conclusão importante:** o fracasso do VDN **não era** só
simetria/config — é evidência de que a **decomposição aditiva `Q1+Q2` é
genuinamente limitada** para este problema (recompensa de equipe + acoplamento
entre robôs). Isso motiva diretamente o **QMIX** (mixer não-linear) como próximo
passo — ver Trabalhos Futuros.
- **MAPPO — não decolou por causa de um BUG, não do algoritmo.** O método
  `MAPPOAgent.select_action` usava **ε-greedy + `argmax`** numa política que é
  **estocástica (PPO)**. O PPO exige **amostrar** da distribuição
  (`Categorical(probs).sample()`) e registrar o `log_prob` *dessa* amostragem;
  em vez disso, o código forçava ações aleatórias/determinísticas mas gravava a
  probabilidade do softmax — o que **invalida a razão de importância** do PPO e
  transforma o gradiente em ruído. Por isso o MAPPO ficou **pior que o aleatório**
  (−9,5 vs +85).

#### ✅ Correção do MAPPO (feita e validada)
Trocando a seleção de ação por amostragem categórica correta:
```python
dist = torch.distributions.Categorical(probs)
action = dist.sample()          # treino: amostra da política
log_prob = dist.log_prob(action)  # log_prob DESSA amostragem
```
(removendo ε-greedy/argmax — a exploração passa a vir da política estocástica +
bônus de entropia), além de aliviar o update (PPO_EPOCHS 20→8, minibatch 16→128,
mais estável), o **MAPPO passou a empatar com o IDQN**: recompensa **313,70**,
**8,00/8 entregas, 100% de sucesso** (ver tabela na Seção 4). Isso **confirma**
que o fracasso original era um erro de implementação, não uma limitação do método.
Script reproduzível: [`codigo_analise/mappo_corrigido.py`](codigo_analise/mappo_corrigido.py)
(roda local em ~5 min: `python mappo_corrigido.py --episodes 3000 --device cpu --ppo-epochs 8 --minibatch 128`).
- **Lição geral:** métodos mais sofisticados (CTDE/on-policy) têm **mais
  hiperparâmetros sensíveis**. Sem tuning cuidadoso, perdem para um IDQN bem
  ajustado **neste cenário de recompensa densa**. Isso **não** significa que VDN
  e MAPPO sejam piores em geral — em tarefas de recompensa esparsa e forte
  acoplamento, a vantagem tende a se inverter.

### O baseline aleatório como sanity check
Random entregar ~3/8 caixas confirma que **o shaping é muito forte**: o ambiente
"empurra" os robôs para as caixas mesmo sem aprender. É um ótimo argumento para
relativizar resultados — a barra de comparação certa é Random, não zero.

---

## 6. ⚠️ Alerta de integridade: os gráficos comparativos atuais são simulados

Os notebooks em
[`_SIMULADO_NAO_USAR/`](_SIMULADO_NAO_USAR/) (`Unifica Colisões`, `Unifica Entregas`,
`Unifica Falha`, `Unifica Taxa de Sucesso`) **não usam os dados reais**. Eles
geram curvas com `np.random.normal(...)` e dicionários de valores digitados à mão.
As curvas inventadas **contradizem os relatórios reais** — por exemplo, mostram a
taxa de sucesso do MAPPO subindo para ~0,4, quando o MAPPO real entrega 0,07/8.

**Não apresente esses gráficos.** Use os de `comparativos/` (Seção 4), que leem
os CSVs reais. Veja `codigo_analise/README.md`.

---

## 7. Trabalhos futuros (novas possibilidades)

1. **QMIX em vez de VDN.** Troca a soma `Q1+Q2` por um **mixer monotônico
   não-linear** condicionado ao estado global. Supera o limite expressivo da
   decomposição aditiva — e é o próximo passo **mais bem justificado**, já que o
   VDN ajustado (Seção 5) mostrou que o limite não era simetria/config, e sim a
   própria soma aditiva.
2. **Parameter sharing.** Como os robôs são homogêneos, compartilhar pesos (com
   um ID de agente na observação) acelera o aprendizado e reduz não-estacionariedade.
3. ~~Re-tuning de VDN~~ **FEITO** (Seção 5): MAPPO corrigido e VDN ajustado já
   foram re-treinados. O VDN ajustado melhorou mas ainda falha → ir para QMIX.
4. **Executar e salvar o HATRPO.** O código existe; falta a corrida completa para
   fechar a comparação dos 5 métodos.
5. **Reward esparso vs. denso (ablation).** Repetir com recompensa só nas entregas.
   É o teste que **realmente** separa coordenação de "seguir o gradiente" — e onde
   CTDE pode finalmente brilhar.
6. **Currículo.** Começar com menos caixas / menos falhas e aumentar a dificuldade
   gradualmente.
7. **Escalar.** Mais robôs (3–4) e mapa maior — aí a coordenação passa a importar
   de verdade e o IDQN tende a sofrer.
8. **Rigor estatístico.** Rodar **múltiplas seeds** e reportar média ± intervalo
   de confiança, em vez de uma única corrida por método.

---

## 8. Mapa rápido dos arquivos

| Arquivo | Conteúdo |
|---------|----------|
| `experimentos/Executa Experimento IDQN.ipynb` | Ambiente canônico + IDQN (o vencedor) |
| `experimentos/Executa Experimento VDN.ipynb` | VDN (`VDNController`) |
| `experimentos/Executa Experimento MPPO.ipynb` | MAPPO |
| `experimentos/Executa Experimento HATRPO.ipynb` | HATRPO (sem run salvo) |
| `experimentos/Executa Experimento Randomico.ipynb` | Baseline aleatório |
| `resultados/<modelo>/` | CSVs, relatórios e PNGs reais de cada corrida |
| `_SIMULADO_NAO_USAR/Unifica*.ipynb` | ⚠️ comparativos **simulados** — não usar |
| `comparativos/` + `codigo_analise/` | ✅ comparativo honesto (script + PNGs + tabela) |
| `codigo_analise/mappo_corrigido.py` | ✅ MAPPO com o bug consertado (empata com IDQN) |
| `codigo_analise/vdn_ajustado.py` | 🔬 VDN com MAX_STEPS=500 + ID do agente (melhora, mas ainda falha) |
| `resultados/mappo_corrigido/`, `resultados/vdn_ajustado/` | resultados reais dos re-treinos |
| `legado/Executa IDQN - Versão 7.0.0.py` | Versão antiga usando a lib externa `rware` (não é o ambiente custom) |
