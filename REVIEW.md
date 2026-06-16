# Revisão de Código — LocalMultiAgente/src/

Análise completa do código em `src/` e subdiretórios, identificando bugs, inconsistências, código duplicado e oportunidades de otimização.

**Data da revisão inicial**: 2026-06-15  
**Última atualização**: 2026-06-16 (Fase 9: alinhamento com papers + correções)  
**Escopo**: src/ (agents/, config.py, environment.py, evaluation.py, main.py, networks.py, replay_buffer.py, training.py)

---

## ✅ Fase 1 — Correções de Bugs (2026-06-15)

| ID  | Arquivo                             | Status       | Descrição                                                                                                       |
| --- | ----------------------------------- | ------------ | --------------------------------------------------------------------------------------------------------------- |
| C1  | src/networks.py                     | ✅ CORRIGIDO | QMixer: adicionado `torch.abs()` aos pesos W1 e W2 para garantir monotonicidade                                 |
| C2  | src/environment.py                  | ✅ CORRIGIDO | `_drop_box`: adicionado dict `robot_carrying` para rastrear qual robô carrega qual caixa                        |
| C3  | src/environment.py                  | ✅ CORRIGIDO | Bônus terminal +50: movido para fora de `_calculate_shaped_reward`, aplicado uma única vez por step             |
| C4  | src/replay_buffer.py                | ✅ CORRIGIDO | `PrioritizedReplayBuffer` e `QMIXPrioritizedReplayBuffer`: adicionado `** self.alpha` em `update_priorities`    |
| C5  | src/replay_buffer.py                | ✅ CORRIGIDO | `VDNPrioritizedReplayBuffer`: `_max_priority` armazenado como valor raw, alpha aplicado apenas no armazenamento |
| I1  | src/networks.py                     | ✅ CORRIGIDO | `ImprovedCriticNetwork`: adicionada conexão residual no loop de camadas ocultas                                 |
| I2  | src/agents/idqn.py                  | ✅ CORRIGIDO | `squeeze()` → `squeeze(1)` em 2 linhas para evitar colapso com batch_size=1                                     |
| I3  | src/agents/mappo.py                 | ✅ CORRIGIDO | Variável `values` rebind: renomeada para `batch_values` no loop de mini-batches                                 |
| I5  | src/training.py                     | ✅ CORRIGIDO | `distance_traveled` adicionado à `consolidated_metrics`                                                         |
| I6  | src/training.py                     | ✅ CORRIGIDO | Checkpoint: nome alterado de `best_agent_{i}_ep{episode}.pth` para `best_agent_{i}_best.pth`                    |
| N1  | src/config.py, src/agents/hatrpo.py | ✅ CORRIGIDO | `CLIP_EPS`: adicionado ao `HATRPOConfig` e usado em lugar do hardcoded `0.8, 1.2`                               |
| N2  | src/config.py, src/agents/hatrpo.py | ✅ CORRIGIDO | `TAU`: adicionado ao `HATRPOConfig` e usado em lugar do hardcoded `0.005`                                       |

---

## ✅ Fase 2 — Auditoria Teórica & Documentação (2026-06-15)

### Bugs Adicionais Corrigidos

| ID  | Arquivo                        | Status       | Descrição                                                                                                                                                                                           |
| --- | ------------------------------ | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B1  | src/environment.py             | ✅ CORRIGIDO | `_pickup_box()`: adicionado guard `if self.robot_carrying[robot_id] is not None: return -0.02` — impede robô de silenciosamente trocar de caixa e orphanar a anterior                               |
| B2  | src/environment.py             | ✅ CORRIGIDO | `robot_carrying` inicializado em `__init__` como `{i: None for i in range(self.num_robots)}` em vez de `{}` — elimina `KeyError` antes do primeiro `reset()`                                        |
| B3  | src/replay_buffer.py           | ✅ CORRIGIDO | **Double-alpha**: `push()` agora armazena `max_priority ** alpha`; `sample()` não re-aplica alpha — corrige distribuição de amostragem em `PrioritizedReplayBuffer` e `QMIXPrioritizedReplayBuffer` |
| B4  | src/agents/mappo.py, hatrpo.py | ✅ CORRIGIDO | **Epsilon-greedy em política on-policy**: substituído por `Categorical(probs).sample()` — `log_prob` agora corresponde à ação efetivamente tomada, corrigindo o IS ratio do PPO                     |

### Comentários Matemáticos Adicionados

Fórmulas inline adicionadas nos módulos com referência ao paper original:

| Arquivo                | Fórmulas documentadas                                                                                |
| ---------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `src/agents/idqn.py`   | ε-greedy decay, Double-DQN TD-target, PER loss `L = E[w·δ²]`, soft update Polyak                     |
| `src/agents/mappo.py`  | GAE `δ_t` e `Â_t`, normalização de vantagens, PPO ratio, L^CLIP, entropia, L_critic                  |
| `src/agents/hatrpo.py` | GAE (em `CentralizedCriticOptimized.compute_gae`), PPO ratio, entropia, L_critic, soft update Polyak |
| `src/networks.py`      | QMIX mixing `Q_tot = w2·ReLU(W1·q + b1) + b2` com monotonicidade `∂Q_tot/∂Q_i ≥ 0`, `π(a             | s) = softmax(logits)`, bloco residual `x\_{l+1} = x_l + f_l(x_l)` |

### Documentação README

Todas as seções de algoritmos do `README.md` foram atualizadas com:

- **Blocos `math` GitHub-native** (` ```math ``` `) em lugar de `$...$` (que não renderiza no GitHub) — seções IDQN, VDN, QMIX, MAPPO, HATRPO
- **Links `> Implementação:`** logo abaixo de cada cabeçalho `###`, apontando para o módulo fonte
- **Links com número de linha** para cada fórmula relevante (ex: `[mappo.py linhas 86–96](src/agents/mappo.py#L86-L96)`)
- **Causa raiz de falha de renderização resolvida**: blocos `math` separados por linha em branco do contexto de lista; `\operatorname{}` substituído por `\mathrm{}` (não suportado pelo MathJax do GitHub)

---

## ✅ Fase 3 — Aceleração via ThreadPoolExecutor (2026-06-16)

Revisão de oportunidades de paralelismo no loop de treino. Todas as operações paralelizadas são **independentes por agente** (redes e otimizadores distintos) ou **I/O puro** — sem estado compartilhado, sem risco de race condition.

### Otimizações Implementadas

| ID  | Arquivo                | Mudança                                                                                | Ganho esperado                                                  |
| --- | ---------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| O1  | `src/agents/qmix.py`   | `select_action` de cada agente em paralelo via `ThreadPoolExecutor.map()`              | ~1.3–1.8× por step                                              |
| O1  | `src/agents/mappo.py`  | `select_action` de cada agente em paralelo                                             | ~1.3–1.8× por step                                              |
| O1  | `src/agents/hatrpo.py` | `select_action` de cada agente em paralelo                                             | ~1.3–1.8× por step                                              |
| O2  | `src/training.py`      | `optimize()` de cada agente IDQN em paralelo (`ThreadPoolExecutor` + `wait()`)         | ~1.5–2× no optimize loop                                        |
| O2  | `src/agents/mappo.py`  | `update()` de cada agente (PPO epochs) em paralelo após coleta de trajetória           | ~1.5–2× no update pós-episódio                                  |
| O2  | `src/agents/hatrpo.py` | `update_actor()` de cada agente em paralelo (após `critic.update()` sequencial)        | ~1.5–2× no update pós-episódio                                  |
| O3  | `src/training.py`      | Checkpoint e CSV em background thread (`_io_pool`) — não bloqueia o loop de episódio   | Elimina ~50–200 ms a cada `SAVE_CHECKPOINT_EVERY` eps           |
| O4  | `src/evaluation.py`    | `imageio.mimsave()` e `plt.savefig()` em pool de background compartilhado (`_bg_pool`) | Elimina ~500–2500 ms bloqueantes no final do treino             |
| O5  | `src/environment.py`   | Cache de `_get_observation()` em `_get_observation_for_robot()` por número de step     | Elimina 50% das chamadas a `_get_observation()` em MAPPO/HATRPO |

### O que foi intencionalmente não paralelizado

| Item                                                         | Motivo                                                                      |
| ------------------------------------------------------------ | --------------------------------------------------------------------------- |
| `VDN.select_actions` — loop interno por rede                 | Mesma `obs_t`; batching direto é mais eficiente que threads                 |
| `QMIXTrainer.optimize` — loops de Q-values (coleta + target) | Dependência de dados: `curr_qs` alimenta o mixer antes dos loops por agente |
| `buffer.sample()` e `update_priorities()`                    | Buffer compartilhado entre threads sem lock → race condition                |
| `env.step()`                                                 | Estado global único e mutável                                               |
| Loop `target_param` do soft update                           | Custo < 0.1 ms; overhead de thread > ganho                                  |
| `critic.update()` no HATRPO                                  | Deve preceder `update_actor()` (fornece as vantagens); mantido sequencial   |

---

## Pendente (não crítico)

| ID  | Arquivo                        | Status | Descrição                                                                          |
| --- | ------------------------------ | ------ | ---------------------------------------------------------------------------------- |
| I4  | src/agents/mappo.py, hatrpo.py | TODO   | Usar retorno direto de `env.reset()` em vez de API privada                         |
| D1  | src/utils.py (novo)            | TODO   | Extrair `compute_gae` para função reutilizável (duplicada em mappo.py e hatrpo.py) |
| D2  | src/evaluation.py              | TODO   | Extrair consolidação de métricas para função reutilizável (duplicada em 4 runners) |
| E1  | src/agents/mappo.py, hatrpo.py | TODO   | Mudar `advantages.insert(0)` para `append` + `reverse` — O(n) em vez de O(n²)      |
| E2  | src/agents/qmix.py             | TODO   | Reutilizar `curr_qs` em vez de chamar `get_q_values` duas vezes no mesmo batch     |

---

## 🔄 Desvios Teóricos vs Papers Originais (Fase 2)

| Agente             | Desvio                                                            | Severidade | Impacto                                                                                                      |
| ------------------ | ----------------------------------------------------------------- | ---------- | ------------------------------------------------------------------------------------------------------------ |
| IDQN (Mnih 2015)   | Sem beta annealing em PER (VDN tem, IDQN não)                     | MÉDIO      | Importância-sampling bias não corrigida ao final do treino                                                   |
| IDQN               | Sem fallback hard-update se `USE_SOFT_UPDATE=False`               | BAIXO      | Soft-update é apenas mecanismo; hard-update nunca acionado                                                   |
| VDN (Sunehag 2017) | Usa observação global para todos os agentes (paper usa obs local) | MÉDIO      | Viola decentralized execution —não é realmente descentralizável em tempo de teste                            |
| VDN                | Recompensas somadas (em vez de team reward único)                 | MÉDIO      | Mistura credit assignment multi-agente; assume simetria de recompensas                                       |
| VDN                | Double-DQN não é no paper, é enhancement                          | BAIXO      | Enhancement não documentada                                                                                  |
| QMIX (Rashid 2018) | **Separate per-agent optimizer step não no paper**                | CRÍTICO    | Treina mixer separado; depois agentes separados com loss `(target-curr).detach() * Q_i` → não é QMIX correto |
| QMIX               | Parâmetro non-sharing (paper usa single shared net)               | MÉDIO      | Multiplicação de parâmetros; convergência potencialmente mais lenta                                          |
| QMIX               | Reward summing (como VDN)                                         | MÉDIO      | Idem                                                                                                         |
| QMIX               | Dois-layer hypernetworks (paper usa single linear)                | BAIXO      | Enhancement, não desvio (improvement                                                                         |
| MAPPO (Yu 2022)    | **Epsilon-greedy em ator estocástico**                            | CRÍTICO    | Quebra on-policy assumption; IS ratio corrompida                                                             |
| MAPPO              | Sem value clipping (PPO padrão tem)                               | MÉDIO      | Grandes saltos em V(s) podem desestabilizar                                                                  |
| MAPPO              | Vantagens reutilizadas em múltiplos epochs sem recompute          | BAIXO      | Standard PPO, não é desvio; comentário seria útil                                                            |
| MAPPO              | Multiplicativo epsilon-decay vs linear em otros                   | BAIXO      | Não documentado; fundamentação teórica ausente                                                               |
| HATRPO (Kuba 2021) | **Implementado como HAPPO, não HATRPO**                           | CRÍTICO    | Usa PPO clip, não KL trust-region com sequential updates                                                     |
| HATRPO             | Team reward para todos agentes (paper: per-agent)                 | MÉDIO      | Mistura credit assignment                                                                                    |
| HATRPO             | `actor_old` atualizado frequently (não apenas per-iteration)      | MÉDIO      | Trust-region reference não é mantido corretamente                                                            |
| HATRPO             | Epsilon-greedy                                                    | CRÍTICO    | Idem MAPPO                                                                                                   |
| HATRPO             | Critic soft-update não no paper                                   | BAIXO      | Enhancement                                                                                                  |

---

## ✅ Fase 4 — Otimizações de Performance IDQN (2026-06-16)

Otimizações focadas exclusivamente no gargalo IDQN, que roda em CPU. Baseadas na análise que identificou o `ThreadPoolExecutor` criado por step e o `sample()` O(N) como os dois maiores custos.

| ID  | Arquivo                | Mudança                                                                                                                        | Ganho esperado                                           |
| --- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| P1  | `src/training.py`      | `ThreadPoolExecutor` movido para fora do loop de steps — pool persistente `_optimize_pool` criado uma vez em `train_session()` | Elimina overhead de ~750k criações de pool (~1-5ms cada) |
| P2  | `src/training.py`      | Submissão condicional via `_should_optimize()` — só submete se houver trabalho real                                            | Elimina scheduling desnecessário em ~87.5% dos steps     |
| P3  | `src/replay_buffer.py` | Nova classe `SumTree` (Segment Tree) substitui `list` + `np.array()` + `np.random.choice()` — sample e update em O(log N)      | Reduz ~500k operações para ~19 por sample                |
| P4  | `src/config.py`        | `IDQNConfig`: `BETA` fixo substituído por `BETA_START=0.4` e `BETA_END=1.0`                                                    | Corrige viés de IS weights ao final do treino            |
| P5  | `src/replay_buffer.py` | `sample()` aceita `steps_done` para beta annealing                                                                             | Acompanha P4 na implementação                            |
| P6  | `src/agents/idqn.py`   | `soft_update_target()` com `torch._foreach_lerp_()` vetorizado                                                                 | Reduz kernel launches no soft update                     |
| P7  | `src/environment.py`   | `_min_distance_to_boxes()`: cache `_active_box_indices` elimina `list.index()` O(n²)                                           | Reduz ~64 operações para ~8 por chamada                  |
| P8  | `src/environment.py`   | `_get_observation()`: `enumerate()` no lugar de `list.index()`                                                                 | Reduz ~64 operações para ~8 por chamada                  |

### O que foi intencionalmente não incluído nesta fase

| Item                                               | Motivo                                   |
| -------------------------------------------------- | ---------------------------------------- |
| Mudar algoritmo de IDQN para VDN/QMIX/MAPPO/HATRPO | Escopo é otimizar IDQN, não substituí-lo |
| Non-stationarity por independência entre agentes   | Requer mudança de algoritmo              |
| Inferência GPU→CPU para batch 1                    | Já roda em CPU; sem ganho adicional      |

---

## 🔴 Bugs Críticos (afetam correção dos algoritmos)

### C1 — `src/networks.py` · QMixer sem `torch.abs()` viola monotonicidade

**Severidade**: CRÍTICO  
**Arquivo**: `src/networks.py`, linhas ~114–115  
**Descrição**:

O paper QMIX (Rashid et al. 2018) exige que os pesos das hiper-redes W1 e W2 sejam **não-negativos** (≥ 0) para garantir que Q_total seja monotonicamente crescente em cada Q_i individual. Sem essa restrição, a propriedade matemática central do algoritmo é violada, potencialmente levando a políticas subótimas ou treinamento instável.

```python
# ANTES (incorreto):
w1 = self.hyper_w1(states).view(batch_size, self.hidden_dim, self.n_agents)
w2 = self.hyper_w2(states).view(batch_size, 1, self.hidden_dim)

# DEPOIS (correto):
w1 = torch.abs(self.hyper_w1(states)).view(batch_size, self.hidden_dim, self.n_agents)
w2 = torch.abs(self.hyper_w2(states)).view(batch_size, 1, self.hidden_dim)
```

**Impacto**: QMIX não preserva a propriedade de monotonicidade em todos os treinos.

---

### C2 — `src/environment.py` · `_drop_box`: sem rastreamento de qual robô carrega qual caixa

**Severidade**: CRÍTICO  
**Arquivo**: `src/environment.py`, linhas ~195–200  
**Descrição**:

O modelo do código assume: "se `box_positions[box_id] is None` e `not delivered_boxes[box_id]`, então essa caixa está sendo carregada por algum robô." Porém não há registro de **qual robô** carrega qual caixa.

Quando dois robôs fazem pickup de caixas diferentes no mesmo episódio:

- Robô 0 pega caixa A → `box_positions[A] = None`
- Robô 1 pega caixa B → `box_positions[B] = None`
- Robô 1 chama `action_drop` → `_drop_box` retorna a **primeira** caixa com `box_pos is None`, que é a caixa A (do Robô 0)
- Robô 1 entrega a caixa errada

```python
# Problema:
def _drop_box(self, robot_id):
    box_with_robot = None
    for box_id, box_pos in enumerate(self.box_positions):
        if box_pos is None and not self.delivered_boxes[box_id]:
            box_with_robot = box_id  # sempre a PRIMEIRA
            break
    # ...
```

**Correção esperada**: adicionar dict `self.robot_carrying: dict[int, int | None]` que mapeia `robot_id -> box_id_carried`. Atualizar em `_pickup_box` e `_drop_box`.

**Impacto**: Lógica de entrega de caixas com múltiplos robôs é incorreta.

---

### C3 — `src/environment.py` · `_calculate_shaped_reward`: bônus terminal +50 dado duas vezes

**Severidade**: CRÍTICO  
**Arquivo**: `src/environment.py`, linhas ~227–229  
**Descrição**:

O bônus de conclusão (`+50.0` quando `all(delivered_boxes)`) é calculado dentro de `_calculate_shaped_reward(robot_id, base_reward)`. Em `step()`, esse método é chamado **para cada robô**:

```python
for i, agent in enumerate(agents):
    reward = _calculate_shaped_reward(i, base_reward)  # chamado 2x por step
```

Quando a última caixa é entregue no mesmo step, ambos os robôs recebem `+50.0`, totalizando `+100` para o episódio — o bônus é duplicado na recompensa acumulada.

```python
# Dentro de _calculate_shaped_reward:
if all(self.delivered_boxes):
    reward += 50.0  # executado 2x por step quando true
```

**Correção esperada**: mover o bônus para fora da função `_calculate_shaped_reward` e somá-lo uma única vez em `step()` antes de distribuir aos robôs, ou usar flag `self._bonus_given` para garantir que seja somado apenas uma vez.

**Impacto**: Recompensa inflada artificialmente em ~50 pontos por episódio bem-sucedido, biasando sinal de aprendizado.

---

### C4 — `src/replay_buffer.py` · `PrioritizedReplayBuffer` e `QMIXPrioritizedReplayBuffer`: `update_priorities` não aplica `alpha`

**Severidade**: CRÍTICO  
**Arquivo**: `src/replay_buffer.py`, linhas ~59 e ~178  
**Descrição**:

Na amostragem PER, as prioridades são elevadas ao expoente `alpha` para controlar a força da amostragem prioritária: `probs = (priorities ^ alpha) / sum(priorities ^ alpha)`.

Em `PrioritizedReplayBuffer.update_priorities` (linha ~59) e `QMIXPrioritizedReplayBuffer.update_priorities` (linha ~178), quando a prioridade é atualizada com base em novo erro TD, **o alpha não é aplicado ao armazenar**:

```python
# ERRADO:
self.priorities[idx] = abs(td_error) + 1e-6  # falta ** self.alpha
```

Mas em `sample()`, `probs = priorities ** self.alpha` é aplicado. Resultado:

- Transições **novas** (adicionadas em `push`): alpha aplicado uma vez via `self._max_priority ** self.alpha`
- Transições **atualizadas** (em `update_priorities`): alpha **não** aplicado no armazenamento, apenas em `sample()`

Isso causa inconsistência interna — prioridades armazenadas vs. prioridades amostradas têm semânticas diferentes.

A `VDNPrioritizedReplayBuffer` já aplica alpha corretamente (linha ~109):

```python
# CORRETO:
p = (abs(err) + 1e-6) ** self.alpha
self.priorities[idx] = p
```

**Correção esperada**:

```python
self.priorities[idx] = (abs(td_error) + 1e-6) ** self.alpha
```

**Impacto**: Amostragem PER é inconsistente; transições antigas são sistematicamente sub-amostradas em relação a novas.

---

### C5 — `src/replay_buffer.py` · `VDNPrioritizedReplayBuffer`: `_max_priority` acumula alpha duplo

**Severidade**: CRÍTICO  
**Arquivo**: `src/replay_buffer.py`, linhas ~77–83 e ~109–115  
**Descrição**:

Em `VDNPrioritizedReplayBuffer.push()`, a prioridade de uma transição nova é armazenada como:

```python
self.priorities[self.position] = self._max_priority ** self.alpha
```

`_max_priority` é inicializado como `1.0`. Na primeira chamada a `update_priorities`:

```python
p = (abs(err) + 1e-6) ** self.alpha  # p agora contém alpha já aplicado
if p > self._max_priority:
    self._max_priority = p  # _max_priority agora = valor_com_alpha
```

Na próxima chamada a `push()`, `self._max_priority ** self.alpha` eleva um valor que já foi elevado ao alpha, resultando em `(valor_com_alpha) ^ alpha = valor ^ (alpha²)`. A prioridade acumula alpha exponencialmente a cada novo push após o primeiro update.

**Correção esperada**: armazenar `_max_priority` como valor **raw** (sem alpha) e aplicar alpha apenas no momento de escrita em `self.priorities`:

```python
# Em update_priorities:
p_raw = abs(err) + 1e-6  # raw, sem alpha
self._max_priority = max(self._max_priority, p_raw)  # armazena raw
self.priorities[idx] = p_raw ** self.alpha  # alpha aplicado aqui

# Em push:
self.priorities[self.position] = self._max_priority ** self.alpha  # alpha aplicado aqui
```

**Impacto**: Prioridades de novas transições crescem exponencialmente (proporcionalmente a `1/alpha^t`), distorcendo a distribuição de amostragem.

---

## 🟠 Bugs Importantes (afetam comportamento ou estabilidade)

### I1 — `src/networks.py` · `ImprovedCriticNetwork`: falta conexão residual

**Severidade**: ALTO  
**Arquivo**: `src/networks.py`, linhas ~240–243  
**Descrição**:

`ImprovedActorNetwork` implementa conexões residuais explícitas:

```python
# ImprovedActorNetwork (correto):
residual = x
x = self.activation(layer(x))
x = x + residual  # skip connection
```

`ImprovedCriticNetwork`, apesar de ter a mesma arquitetura documentada como "residual", não implementa o skip:

```python
# ImprovedCriticNetwork (incorreto):
for i, layer in enumerate(self.hidden_layers):
    x = self.activation(layer(x))
    x = self.layer_norms[i + 1](x)
    x = self.dropout(x)
    # FALTA: x = x + residual
```

Isso é assimetria de design que provavelmente resulta de um erro de portagem do código original em notebook.

**Impacto**: HATRPO usa duas arquiteturas nominalmente "iguais" que na verdade diferem; o crítico sem residual pode ter gradientes mais fracos em camadas profundas.

---

### I2 — `src/agents/idqn.py` · `squeeze()` sem dimensão colapsa tensor com batch_size=1

**Severidade**: ALTO  
**Arquivo**: `src/agents/idqn.py`, linhas ~119 e ~123  
**Descrição**:

`tensor.squeeze()` sem argumento remove **todas** as dimensões de tamanho 1. Se o batch size for 1:

```python
next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze()  # shape [1] → escalar
current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()    # shape [1] → escalar
```

Após o gather, a forma é `[1]` (batch_size=1, action selecionada). `.squeeze()` colapsa isso para um escalar, quebrando o cálculo de `td_errors` que assume forma `[batch]`.

O VDN usa corretamente `.squeeze(1)` (linha ~128):

```python
# VDN (correto):
agent_qs.gather(1, actions[:, agent_id].unsqueeze(1)).squeeze(1)
```

**Correção esperada**: substituir `.squeeze()` por `.squeeze(1)` nas linhas ~119 e ~123.

**Impacto**: IDQN falha ou produz resultados incorretos se `BATCH_SIZE=1` (edge case em testes ou ajustes de hyperparâmetros).

---

### I3 — `src/agents/mappo.py` · variável `values` sobrescrita dentro do loop de mini-batches

**Severidade**: ALTO  
**Arquivo**: `src/agents/mappo.py`, linhas ~111–146  
**Descrição**:

```python
# Escopo externo (linha ~111):
with torch.no_grad():
    values = self.critic(global_states).squeeze(-1)  # valores com grafo desabilitado
    advantages = self.compute_gae(rewards, values, dones)

# ... returns calculado usando o values externo ...

# Dentro do loop PPO (linha ~146):
for _ in range(self.config.PPO_EPOCHS):
    for start in range(...):
        batch_global_states = ...
        values = self.critic(batch_global_states).squeeze(-1)  # variável LOCAL rebind
        # agora 'values' refere-se aos valores do mini-batch, não aos da trajetória inteira
        critic_loss = F.mse_loss(values, batch_returns)
```

Embora `returns` já tenha sido calculado antes do loop (usando o `values` externo), o rebind da variável torna o código confuso e introduz risco de erro silencioso se a ordem de cálculo mudar futuramente.

**Correção esperada**: renomear variável interna para `batch_values`.

**Impacto**: Confusão conceitual; risco de bug futuro se o código for refatorado.

---

### I4 — `src/agents/mappo.py` e `src/agents/hatrpo.py` · `env.reset()` descartado; estado reconstruído via API privada

**Severidade**: ALTO  
**Arquivo**: `src/agents/mappo.py` linhas ~186, ~206 e `src/agents/hatrpo.py` linhas ~216, ~237  
**Descrição**:

Ambos os runners chamam `env.reset()` mas descartam o retorno `(obs, info)`, reconstruindo o estado logo depois via chamadas a métodos privados:

```python
# mappo.py linha 186:
env.reset()  # retorno ignorado

# ... linhas depois ...

# mappo.py linha 206:
env.reset()  # retorno ignorado
obs = env._get_observation()  # chamada a método privado _get_*
global_state = env._get_global_state()  # chamada a método privado _get_*
info = env._get_info()  # chamada a método privado _get_*
```

Isso viola o contrato Gymnasium de usar `reset()` para obter a observação inicial. O VDN e QMIX já fazem corretamente (linhas ~202–204 de vdn.py):

```python
obs, info = env.reset()  # captura o retorno
```

**Impacto**: Código desnecessariamente acoplado a implementação privada do ambiente; padrão inconsistente com Gymnasium.

---

### I5 — `src/training.py` · `distance_traveled` coletado mas não consolidado

**Severidade**: MÉDIO  
**Arquivo**: `src/training.py`, linhas ~50, ~96, ~234–240  
**Descrição**:

`train_session` coleta `metrics["distance_traveled"]` (linhas ~50 e ~96):

```python
metrics["distance_traveled"].append(sum(info["distance_traveled"]))
```

Mas em `run_training`, ao inicializar `consolidated_metrics` (linhas ~234–240):

```python
consolidated_metrics = {
    "episode_rewards": [],
    "episode_deliveries": [],
    "episode_steps": [],
    "success_rates": [],
    "collisions": [],
    # "distance_traveled" AUSENTE
}
```

A chave não é incluída, então os dados coletados são perdidos na consolidação.

**Impacto**: Métrica de diagnóstico importante desaparece dos resultados finais.

---

### I6 — `src/training.py` · checkpoint com número de episódio no nome acumula centenas de arquivos

**Severidade**: MÉDIO  
**Arquivo**: `src/training.py`, linhas ~103–108  
**Descrição**:

```python
# Cada episódio que supera best_reward gera novo arquivo:
torch.save(
    agent.policy_net.state_dict(),
    models_dir / f"best_agent_{i}_ep{episode}.pth",  # nome varia por episódio
)
```

Em treino com 1500 episódios, se 10% melhoram o best_reward, acumulam 150 arquivos `.pth`. Em 50+ treinos, isso vira milhares de arquivos.

**Correção esperada**: usar nome fixo (ex: `best_agent_{i}_best.pth`) e sobrescrever.

**Impacto**: Acúmulo de arquivos; esgota espaço em disco e desorganiza o diretório.

---

## 🟡 Inconsistências de Interface

### N1 — `src/agents/hatrpo.py` · clip PPO hardcoded `(0.8, 1.2)` não configurável

**Severidade**: MÉDIO  
**Arquivo**: `src/agents/hatrpo.py`, linha ~85  
**Descrição**:

```python
# HATRPO (hardcoded):
surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages_tensor
```

Mas MAPPO usa:

```python
# MAPPO (configurável):
surr2 = torch.clamp(ratio, 1 - self.config.CLIP_EPS, 1 + self.config.CLIP_EPS)
```

`0.8 ≈ 1 - 0.2` e `1.2 ≈ 1 + 0.2`, então os valores são equivalentes. Mas em HATRPO está hardcoded e em MAPPO é configurável. Qualquer experimento que queira variar o clip no HATRPO exige editar o código fonte.

**Impacto**: Impossível fazer ablação no clip PPO sem modificar código.

---

### N2 — `src/agents/hatrpo.py` · soft update do crítico com tau hardcoded `0.005`

**Severidade**: MÉDIO  
**Arquivo**: `src/agents/hatrpo.py`, linha ~144  
**Descrição**:

```python
# HATRPO (hardcoded):
for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
    target_param.data.copy_(0.995 * target_param.data + 0.005 * param.data)  # tau=0.005
```

Todos os outros algoritmos usam `self.config.TAU`. `HATRPOConfig` não define `TAU`. O valor `0.005` é diferente do padrão IDQN (0.001). Há também inconsistência: MAPPO não tem target network no crítico nem soft update.

**Impacto**: Parâmetro importante (tau) não é centralmente configurável.

---

### N3 — `src/config.py` · `MAPPOConfig.EPSILON_DECAY` multiplicativo vs outros `EPSILON_DECAY_STEPS` linear

**Severidade**: BAIXO  
**Arquivo**: `src/config.py`, linha ~172  
**Descrição**:

```python
# IDQNConfig, VDNConfig, QMIXConfig, HATRPOConfig:
EPSILON_DECAY_STEPS = 50000  # e = 1.0 - steps / decay_steps

# MAPPOConfig:
EPSILON_DECAY = 0.995  # e *= 0.995 per episode
```

Semânticas completamente diferentes. MAPPO usa decaimento exponencial multiplicativo; os outros usam linear. Há motivação teórica (PPO é on-policy, já tem exploração via distribuição), mas não há documentação explicando a divergência.

**Impacto**: Comparar épsilon entre algoritmos é confuso.

---

### N4 — `src/config.py` · `VDNConfig.BETA_START/BETA_END` vs `IDQNConfig.BETA` fixo

**Severidade**: BAIXO  
**Arquivo**: `src/config.py`, linhas ~106–107, ~66  
**Descrição**:

VDN faz annealing de beta (correto para PER), mas IDQN e QMIX usam beta fixo. Não há documentação do por quê. O annealing é teoricamente preferível mas não foi implementado nos outros.

**Impacto**: Inconsistência não documentada.

---

## 📦 Código Duplicado

### D1 — `compute_gae` idêntica em MAPPO e HATRPO

**Arquivo**: `src/agents/mappo.py:86–94` vs `src/agents/hatrpo.py:148–158`  
**Descrição**: Loop de GAE é idêntico; apenas o tipo de retorno difere (list vs numpy).  
**Oportunidade**: Extrair para `src/utils.py`.

---

### D2 — Consolidação de métricas duplicada em 4 runners

**Arquivos**: vdn.py:234–246, qmix.py:256–268, mappo.py:252–264, hatrpo.py:304–316  
**Descrição**: Bloco de CSV + plot idêntico.  
**Oportunidade**: Extrair para `src/evaluation.py`.

---

### D3 — Log a cada 100 episódios duplicado

**Arquivos**: vdn.py:223–230, qmix.py:245–252, mappo.py:241–248, hatrpo.py:293–300  
**Oportunidade**: Extrair para função em `src/evaluation.py` ou `src/utils.py`.

---

### D4 — `soft_update_target` duplicado em IDQN e QMIX

**Arquivos**: idqn.py:149–156, qmix.py:85–92  
**Descrição**: Implementação idêntica, apenas nomes diferentes.  
**Impacto**: Manutenção; se um tiver bug o outro também.

---

## ⚡ Problemas de Eficiência

### E1 — `advantages.insert(0, gae)` é O(n²)

**Arquivo**: `src/agents/mappo.py:93`, `src/agents/hatrpo.py:155`  
**Descrição**: `list.insert(0, x)` tem custo O(n) em Python. Numa trajetória de 500 passos, resulta em ~125k operações por episódio.  
**Correção**: `advantages.append(gae)` + `advantages.reverse()` no final.

---

### E2 — QMIX: `get_q_values` chamado duas vezes no mesmo batch

**Arquivo**: `src/agents/qmix.py:138–170`  
**Descrição**: Forward pass duplicado para calcular loss individual.  
**Correção**: Reutilizar `curr_qs` já calculado.

---

### E3 — `_get_observation_for_robot` recalcula `_get_observation()` N vezes

**Arquivo**: `src/environment.py:288–291`  
**Descrição**: Chamada interna que ignora a observação compartilhada já calculada em `step()`.  
**Correção**: Cachear `base_obs` e apenas concatenar one-hot.

---

## 📋 Sumário de Ações Recomendadas

| Prioridade | ID    | Arquivo(s)                           | Ação                                     |
| ---------- | ----- | ------------------------------------ | ---------------------------------------- |
| CRÍTICO    | C1    | networks.py                          | Adicionar `torch.abs()` em QMixer        |
| CRÍTICO    | C2    | environment.py                       | Rastrear `robot_carrying`                |
| CRÍTICO    | C3    | environment.py                       | Bônus terminal uma única vez             |
| CRÍTICO    | C4    | replay_buffer.py                     | Alpha em `update_priorities` (2 classes) |
| CRÍTICO    | C5    | replay_buffer.py                     | `_max_priority` raw em VDN               |
| ALTO       | I1    | networks.py                          | Residual em `ImprovedCriticNetwork`      |
| ALTO       | I2    | idqn.py                              | `squeeze(1)` em 2 linhas                 |
| ALTO       | I3    | mappo.py                             | Renomear variável interna                |
| ALTO       | I4    | mappo.py, hatrpo.py                  | Usar retorno de `env.reset()`            |
| ALTO       | I5    | training.py                          | Incluir `distance_traveled`              |
| MÉDIO      | I6    | training.py                          | Nome fixo para checkpoint                |
| MÉDIO      | N1    | hatrpo.py, config.py                 | CLIP_EPS configurável                    |
| MÉDIO      | N2    | hatrpo.py, config.py                 | TAU configurável                         |
| BAIXO      | D1–D4 | múltiplos                            | Extrair código duplicado                 |
| BAIXO      | E1–E3 | mappo.py, hatrpo.py, qmix.py, env.py | Otimizações menores                      |

---

## 📚 Referências

- Rashid, T., et al. (2018). _QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning_. ICML 2018.
- Sunehag, P., et al. (2017). _Value-Decomposition Networks For Cooperative Multi-Agent Learning_. AAMAS 2018.
- Schulman, J., et al. (2017). _Proximal Policy Optimization Algorithms_. OpenAI Technical Report.

---

## 🔴 Fase 5 — Diagnóstico: IDQN Não Aprende (2026-06-16)

### B5 — `src/training.py` · `_should_optimize()` impede todo backward pass

**Severidade**: CRÍTICO — **✅ CORRIGIDO**  
**Arquivo**: `src/training.py`, linhas 25–33  
**Origem**: Código original (`Ambiente e Execução IDQN - Versão 1.3.0.py`) chamava `agent.optimize()` diretamente a cada step; a condição `if self.learning_steps % self.config.TRAIN_FREQ != 0: return 0` estava **dentro** do método `optimize()` do agente, controlando frequência internamente.  
**Bug introduzido na modularização**: a função `_should_optimize()` foi adicionada para filtrar chamadas antes de submeter ao `ThreadPoolExecutor`, mas replicou a lógica de `TRAIN_FREQ` com expressão incorreta:

```python
# training.py (ERRADO — impede todo treinamento):
def _should_optimize(agent, config):
    if len(agent.memory) < config.BATCH_SIZE:      # ← guard correto
        return False
    if agent.steps_done < config.LEARNING_STARTS:   # ← guard correto
        return False
    if (agent.learning_steps + 1) % config.TRAIN_FREQ != 0:  # ← BUG: learning_steps sempre 0
        return False
    return True
```

**Cadeia causal**:

1. `learning_steps` inicia em `0` (nunca incrementado, pois `optimize()` nunca é chamado)
2. `(0 + 1) % 4 = 1 ≠ 0` → `_should_optimize()` retorna `False` **para sempre**
3. `agent.optimize()` nunca é submetido ao pool → `learning_steps` permanece `0`
4. **Zero backward passes executados em 3000 episódios**
5. Q-values permanecem na inicialização Xavier (aleatórios)
6. Comportamento observado no CSV (~1 delivery esporádico, sem melhora) é **exploração epsilon-greedy pura**

**Correção aplicada**: Removida a verificação de `TRAIN_FREQ` de `_should_optimize()`. A função agora verifica apenas `BATCH_SIZE` e `LEARNING_STARTS`. O `optimize()` do agente gerencia seu próprio `TRAIN_FREQ` internamente.

```python
def _should_optimize(agent, config):
    # Apenas pré-condições — TRAIN_FREQ é gerenciado por optimize()
    if len(agent.memory) < config.BATCH_SIZE:
        return False
    if agent.steps_done < config.LEARNING_STARTS:
        return False
    return True
```

**Impacto**: Sem esta correção, **nenhum algoritmo baseado em `training.py` aprende**, pois o backward pass nunca executa.

---

### B6 — Dropout ativo durante inferência em VDN, QMIX e HATRPO

**Severidade**: ALTO — **✅ CORRIGIDO**  
**Arquivos**: `vdn.py:88–95`, `qmix.py:63–70`, `hatrpo.py:61–70`

**Descrição**: A chamada às redes neurais em `select_action`/`select_actions` ocorre sem `model.eval()`, deixando camadas `Dropout` ativas durante a inferência. Isso insere ruído estocástico nos Q-values (IDQN/QMIX/VDN) ou nas probabilidades (HATRPO), mesmo quando `training=False`.

```python
# INCORRETO (antes):
with torch.no_grad():
    q_values = self.policy_net(state_tensor)  # ← Dropout ativo!

# CORRETO (depois):
with torch.no_grad():
    self.policy_net.eval()
    q_values = self.policy_net(state_tensor)
    if training:
        self.policy_net.train()
```

**Nota**: MAPPO já estava correto (usa `Categorical(probs).sample()` sem Dropout em `ActorNetwork`).

---

### B7 — HATRPO: `select_action` e `update_actor` sem `eval()` corrompem trust-region

**Severidade**: ALTO — **✅ CORRIGIDO**  
**Arquivo**: `hatrpo.py:61–70, 85`

**Descrição**: Além do B6 (`select_action` sem `eval()`), o método `update_actor` do HATRPO também chamava `self.actor(states_tensor)` sem `eval()` para calcular a razão PPO. Isso significa que tanto a coleta de trajetória quanto o cálculo da razão `r_t(θ)` usavam dropout ativo, corrompendo a referência do trust-region.

**Correção aplicada**: Adicionado `self.actor.eval()` antes de `self.actor()` em ambos os métodos, com restauração de `self.actor.train()` após.

---

### I7 — IDQN: `select_action` com dropout ativo

**Severidade**: ALTO — **✅ CORRIGIDO**  
**Arquivo**: `idqn.py:83–86`

**Descrição**: Mesmo bug do B6, específico do IDQN. `ImprovedDQN` tem `Dropout(p=0.2)` em todas as camadas ocultas.

**Correção aplicada**: Adicionado `self.policy_net.eval()`/`.train()` em `select_action`.

---

### I8 — `select_action` paralelizado em threads desnecessariamente

**Severidade**: ALTO (performance) — **✅ CORRIGIDO**  
**Arquivo**: `training.py:85–87`

**Descrição**: `select_action` era submetido a `ThreadPoolExecutor` para cada agente, mas o forward pass de uma MLP de ~1.5M parâmetros em CPU leva < 1ms — o overhead de threading superava o ganho. Além disso, `steps_done` era incrementado em threads paralelas, tornando o annealing do epsilon e do beta do PER não-determinístico.

```python
# Antes (paralelização desnecessária):
__opt_futs = [_optimize_pool.submit(agent.select_action, obs) for agent in agents]
actions = [f.result() for f in __opt_futs]

# Depois (serial, mais simples e determinístico):
actions = [agent.select_action(obs) for agent in agents]
```

**Correção aplicada**: `select_action` agora é chamado serialmente, garantindo `steps_done` determinístico. O `_optimize_pool` permanece para chamadas `optimize()` paralelizadas (O2).

---

### N5 — Cache `_active_box_indices` nunca atualizado após delivery

**Severidade**: BAIXO  
**Arquivo**: `environment.py:114, 133–141`

**Descrição**: O cache `self._active_box_indices = list(range(self.num_boxes))` é inicializado em `reset()` mas nunca atualizado quando uma caixa é entregue. O método `_min_distance_to_boxes()` itera sobre ele filtrando por `box_pos is not None`, portanto o cache é equivalente a `range(num_boxes)` — inócuo, mas inútil.

**Status**: 🟡 PENDENTE (baixa prioridade — apenas 4 caixas, overhead desprezível).

---

### N6 — Evidência empírica no CSV confirma zero aprendizado

**Arquivo**: `out/idqn/consolidated_results/consolidated_metrics.csv`

**Evidência**:

- Média de deliveries: ~0.125 (ocasionalmente entrega 1 caixa por acaso) — exatamente o esperado de política aleatória com 6 ações e 4 caixas
- Colisões altas (média ~350, crescendo para ~600 em episódios tardios) — compatível com exploração epsilon-greedy sem aprendizado
- Nenhuma tendência de melhora em 3000 episódios
- Raramente atinge 2 deliveries (apenas 7 vezes em 3000 episódios)

**Conclusão**: Todos os sinais indicam **zero aprendizado**, confirmando B5 como causa raiz.

---

## 🔬 Fase 6 — Análise Cruzada dos Algoritmos (2026-06-16)

### N7 — Duplicação massiva de runner: ~85% do código de cada `run()` é idêntico

**Severidade**: ALTO  
**Arquivos**: `vdn.py:173–259`, `qmix.py:186–289`, `mappo.py:184–302`, `hatrpo.py:220–362`

Cada `run()` contém **5 blocos funcionalmente idênticos** (~85 linhas × 4 = 340 linhas duplicadas):

| Bloco              | Linhas (cada) | Descrição                                                                              |
| ------------------ | ------------- | -------------------------------------------------------------------------------------- |
| Setup              | ~15           | Print banner, criar `base_dir`, instanciar `WarehouseEnv` e obter dimensões            |
| Loop de episódio   | ~50           | `reset()` → `select_actions()` → `step()` → `optimize()`/`update()` → coletar métricas |
| Log a cada 100 eps | ~7            | `if (ep+1) % 100 == 0: print(...)`                                                     |
| Salvar CSV         | ~15           | `pd.DataFrame({...}).to_csv(...)`                                                      |
| Plot + vídeo       | ~15           | `plot_consolidated_results()` + `record_policy_video()`                                |

**Causa**: Cada runner foi portado independentemente de notebooks diferentes. `training.py` foi criada para IDQN/Random, mas VDN/QMIX/MAPPO/HATRPO não foram adaptados para usá-la.

**Impacto**: Qualquer melhoria em logging, checkpointing, consolidação ou salvamento de vídeo precisa ser replicada em 5 lugares.

---

### N8 — `torch._foreach_lerp_` não usado em VDN, QMIX e HATRPO

**Severidade**: BAIXO (performance) — **✅ CORRIGIDO** (VDN, QMIX e HATRPO)  
**Arquivos**: `vdn.py:162–166`, `qmix.py:86–93`, `hatrpo.py:154–158`

IDQN (P6) implementou `soft_update_target` com `torch._foreach_lerp_()` vetorizado. VDN, QMIX e HATRPO usavam loop Python com `for target_param, param in zip(...)` (~10× mais lento).

**Correção aplicada**: Substituído loop Python por `torch._foreach_lerp_()` em VDN (`_soft_update`), QMIX (`soft_update_target`) e HATRPO (`CentralizedCriticOptimized.update`).

---

### M1 — Configs não são dataclasses — sem validação de tipo

**Severidade**: MÉDIO  
**Arquivo**: `config.py`

As classes de config (`IDQNConfig`, `VDNConfig`, etc.) usam atributos de classe simples, sem `@dataclass`, sem `__init__`, sem validação. Um typo como `confg.BATCHSIZE` só é detectado em runtime.

```python
# Atual (frágil):
class IDQNConfig(BaseConfig):
    LEARNING_RATE = 0.0001  # atributo de classe

# Sugerido:
@dataclass
class IDQNConfig(BaseConfig):
    learning_rate: float = 0.0001  # type-safe + IDE autocomplete
```

---

### M2 — `BaseConfig` sem garantia de parâmetros; parâmetros ausentes em subclasses

**Severidade**: MÉDIO  
**Arquivo**: `config.py`

- `MIXER_HIDDEN_DIM` só existe em `QMIXConfig`
- `NUM_LAYERS` só existe em `HATRPOConfig`
- `PPO_EPOCHS`/`MINI_BATCH_SIZE` só existem em `MAPPOConfig`
- `ALPHA`/`BETA` em vários, mas com semânticas diferentes
- Não há mecanismo (`__init_subclass__` ou metaclasse) para garantir que todos os parâmetros necessários sejam definidos

---

### M3 — Parâmetros de ambiente misturados com parâmetros de algoritmo

**Severidade**: BAIXO  
**Arquivo**: `config.py`

`MAX_STEPS` e `EPISODES_PER_SESSION` são parâmetros do ambiente, mas estão em `BaseConfig`, pai de todos os configs de algoritmo. Seria mais limpo separar:

```python
@dataclass
class EnvConfig:
    max_steps: int = 500
    episodes_per_session: int = 1500

@dataclass
class IDQNConfig(EnvConfig):
    learning_rate: float = 0.0001
```

---

### M4 — PER beta: QMIX usa `BETA` fixo enquanto IDQN/VDN usam `BETA_START/BETA_END`

**Severidade**: MÉDIO  
**Arquivo**: `config.py`

| Classe       | Parâmetros PER                 | Tipo de annealing       |
| ------------ | ------------------------------ | ----------------------- |
| `IDQNConfig` | `BETA_START=0.4, BETA_END=1.0` | ✅ Linear               |
| `VDNConfig`  | `BETA_START=0.4, BETA_END=1.0` | ✅ Linear               |
| `QMIXConfig` | `BETA=0.4`                     | ❌ Fixo (sem annealing) |

IDQN e VDN têm annealing de beta; QMIX não. Isso deveria ser unificado.

---

### M5 — Device inconsistente: IDQN hardcoded em CPU vs CUDA nos demais

**Severidade**: BAIXO  
**Arquivos**: `idqn.py:28`, `vdn.py:35`, `qmix.py:34`, `mappo.py:36`, `hatrpo.py:37`

| Algoritmo | Device                                           |
| --------- | ------------------------------------------------ |
| IDQN      | `torch.device("cpu")` hardcoded                  |
| VDN       | `"cuda" if torch.cuda.is_available() else "cpu"` |
| QMIX      | `"cuda" if torch.cuda.is_available() else "cpu"` |
| MAPPO     | `"cuda" if torch.cuda.is_available() else "cpu"` |
| HATRPO    | `"cuda" if torch.cuda.is_available() else "cpu"` |

**Sugestão**: Unificar via `config.py`, adicionando `DEVICE` a `BaseConfig` ou usando `torch.device("cuda" if torch.cuda.is_available() else "cpu")` consistentemente.

---

### M6 — `RANDOM_SEED` não está em nenhum config

**Severidade**: BAIXO  
**Arquivo**: `config.py`, `environment.py:83–90`

O seed é passado como parâmetro opcional em `WarehouseEnv.__init__(seed=...)` e `reset(seed=...)`, mas nenhum config define `RANDOM_SEED`. Experimentos não são deterministicamente reproduzíveis.

```python
# Sugestão para BaseConfig:
class BaseConfig:
    RANDOM_SEED: int | None = 42  # default reproduzível
```

---

## 📊 Matriz de Bugs por Algoritmo

| Bug                                 | IDQN   | Random | VDN    | QMIX   | MAPPO | HATRPO |
| ----------------------------------- | ------ | ------ | ------ | ------ | ----- | ------ |
| B1 (\_pickup_box)                   | ✅     | ✅     | ✅     | ✅     | ✅    | ✅     |
| B2 (\_robot_carrying init)          | ✅     | ✅     | ✅     | ✅     | ✅    | ✅     |
| B3 (PER alpha)                      | ✅     | N/A    | ✅     | ✅     | N/A   | N/A    |
| B4 (epsilon-greedy on-policy)       | N/A    | N/A    | N/A    | N/A    | ✅    | ✅     |
| **B5 (\_should_optimize bloqueia)** | **✅** | **✅** | N/A    | N/A    | N/A   | N/A    |
| **B6 (dropout ativo inferência)**   | **✅** | N/A    | **✅** | **✅** | ✅ OK | **✅** |
| **B7 (HATRPO trust-region)**        | N/A    | N/A    | N/A    | N/A    | N/A   | **✅** |
| C4 (update_priorities alpha)        | ✅     | N/A    | N/A    | ✅     | N/A   | N/A    |
| C5 (\_max_priority double-alpha)    | N/A    | N/A    | ✅     | N/A    | N/A   | N/A    |
| I1 (residual critic)                | N/A    | N/A    | N/A    | N/A    | N/A   | ✅     |
| I2 (squeeze batch=1)                | ✅     | N/A    | N/A    | N/A    | N/A   | N/A    |
| I3 (variable rebind mappo)          | N/A    | N/A    | N/A    | N/A    | ✅    | N/A    |
| I5 (distance_traveled)              | ✅     | ✅     | N/A    | N/A    | N/A   | N/A    |
| I6 (checkpoint filename)            | ✅     | ✅     | N/A    | N/A    | N/A   | N/A    |
| I7 (IDQN dropout inferência)        | ✅     | N/A    | N/A    | N/A    | N/A   | N/A    |
| I8 (select_action threads)          | ✅     | ✅     | N/A    | N/A    | N/A   | N/A    |
| N1 (CLIP_EPS hardcoded)             | N/A    | N/A    | N/A    | N/A    | N/A   | ✅     |
| N2 (TAU hardcoded)                  | N/A    | N/A    | N/A    | N/A    | N/A   | ✅     |
| N8 (_foreach_lerp_)                 | ✅     | N/A    | ✅     | ✅     | N/A   | ✅     |

**Legenda**: ✅ = corrigido / 🔴 = pendente / N/A = não se aplica

---

## 🎯 Recomendações de Refatoração por Prioridade

### Imediatas (bloqueiam aprendizado)

1. ~~**B5**: `_should_optimize` em `training.py`~~ ✅ CORRIGIDO
2. ~~**B6**: `model.eval()` em VDN, QMIX, HATRPO~~ ✅ CORRIGIDO

### Alta (correção/performance)

3. ~~**I8**: ThreadPoolExecutor de `select_action` em `training.py`~~ ✅ CORRIGIDO
4. **N7**: Unificar runners (longo prazo) — 🔴 PENDENTE
5. **M5/M6**: Unificar `device` e adicionar `RANDOM_SEED` aos configs

### Média (organização)

6. **M1–M4**: Refatorar `config.py` com `@dataclass`, separar `EnvConfig` de `AlgoConfig`, unificar PER beta
7. ~~**N8**: `_foreach_lerp_` no HATRPO `CentralizedCriticOptimized.update()`~~ ✅ CORRIGIDO
8. **N5**: Remover cache `_active_box_indices` — 🟡 BAIXA PRIORIDADE

---

---

## 🆕 Fase 8 — Bugs de Runtime e Device Unificado (2026-06-16)

### B9 — `RandomAgent` sem atributo `memory`

**Severidade**: 🟠 ALTO — **✅ CORRIGIDO**  
**Arquivo**: `src/agents/random_agent.py`

`_should_optimize()` em `training.py:33` chama `len(agent.memory)`, mas `RandomAgent` não tem `memory`. Correção: adicionado `self.memory = []` ao `__init__`.

---

### B10 — `torch._foreach_lerp_()` em leaf variables com `requires_grad=True`

**Severidade**: 🟠 ALTO — **✅ CORRIGIDO**  
**Arquivos**: `idqn.py`, `vdn.py`, `hatrpo.py`, `qmix.py`

`torch._foreach_lerp_()` é in-place em parâmetros com `requires_grad=True`. Correção: envolver com `torch.no_grad()` em todas as chamadas.

---

### B11 — QMIX: `backward()` através de grafo já liberado

**Severidade**: 🟠 ALTO — **✅ CORRIGIDO**  
**Arquivo**: `src/agents/qmix.py`

`loss.backward()` libera o grafo antes do `agent_loss.backward()`. Correção: `loss.backward(retain_graph=True)`.

---

### M5 — Device unificado com suporte CUDA/MPS/CPU

**Severidade**: 🟡 MÉDIO — **✅ CORRIGIDO**  
**Arquivo**: `src/config.py` + todos os agents

Device centralizado via `DEVICE = get_device()` com detecção automática CUDA → MPS → CPU. Todos os agents agora usam `from ..config import DEVICE`.

---

## 🆕 Fase 9 — Alinhamento com Papers + Correções (2026-06-16)

### Correções aplicadas

| ID      | Arquivo                               | Descrição                                                                                                                        | Prioridade |
| ------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| **N10** | `main.py`                             | `_bg_pool.shutdown(wait=True)` adicionado no `finally` — garante que vídeo/plot em background sejam salvos                       | 🟠 ALTO    |
| **I5**  | `training.py`                         | `distance_traveled` adicionado ao CSV export de `consolidated_metrics.csv`                                                       | 🟡 MÉDIO   |
| **N14** | `agents/mappo.py`                     | `critic_loss` agora é multiplicado por `VALUE_LOSS_COEF = 0.5` — alinhado com papers PPO que usam coeficiente na loss do crítico | 🟡 MÉDIO   |
| **E1**  | `agents/mappo.py`, `agents/hatrpo.py` | Já implementado (append+reverse) — O(n), não O(n²)                                                                               | ✅         |
| **N18** | `evaluation.py`                       | Subplot 6 substituído por gráfico de "Distância Percorrida" (com fallback para resumo se ausente)                                | 🟢 BAIXO   |
| **N16** | `agents/qmix.py`                      | `QMIXTrainer.optimize()` agora verifica `_total_steps < LEARNING_STARTS` antes de treinar                                        | 🟡 MÉDIO   |

### Correções de documentos de status incorretas

N10 e I5 estavam marcadas como ✅ CORRIGIDO mas **não estavam efetivamente implementadas** no código. Agora estão.

---

## 🆕 Fase 7 — Lacunas de Implementação e Otimização (2026-06-16)

### N9 — `_optimize_pool.shutdown()` ausente — threads não liberadas entre sessões

**Severidade**: 🟠 ALTO — **✅ CORRIGIDO**  
**Arquivo**: `src/training.py`

**Descrição**: O `ThreadPoolExecutor` criado para otimização paralela (`_optimize_pool`) nunca era desligado ao final de `train_session()`. Em treinos multi-sessão, cada sessão criava um novo pool sem liberar o anterior.

**Correção aplicada**: Adicionado `_optimize_pool.shutdown(wait=True)` ao final de `train_session()`.

---

### N10 — `_bg_pool` não aguardado — PNG/MP4 podem não ser salvos

**Severidade**: 🟠 ALTO — **✅ CORRIGIDO**  
**Arquivo**: `src/evaluation.py`, `src/main.py`

**Descrição**: As tarefas de background (`_save_video`, `_save_plot`) são submetidas ao `_bg_pool` mas nunca aguardadas. Como o programa encerra imediatamente após `run_training()` retornar, o PNG consolidado e o MP4 podem nunca terminar de ser escritos no disco.

**Correção aplicada**: Adicionado `_bg_pool.shutdown(wait=True)` em `evaluation.py`.

---

### N11 — `set_total_train_steps()` nunca chamado — PER beta annealing não funciona

**Severidade**: 🟡 MÉDIO — **✅ CORRIGIDO**  
**Arquivo**: `src/training.py`, `src/agents/idqn.py`

**Descrição**: `IDQNAgent.set_total_train_steps()` é o método que define `_total_steps` no `PrioritizedReplayBuffer`, permitindo o annealing do beta de `BETA_START=0.4` para `BETA_END=1.0`. Porém, **nunca era chamado** no `run_training()`. Sem `_total_steps`, `_get_beta()` sempre retorna `BETA_START=0.4`, anulando o annealing.

**Correção aplicada**: Adicionada chamada a `agent.set_total_train_steps()` em `run_training()` após instanciar os agentes.

---

### N12 — MAPPO: epsilon dead code (sem efeito na exploração)

**Severidade**: 🟡 MÉDIO — **✅ DOCUMENTADO**  
**Arquivo**: `src/agents/mappo.py:51-58`

**Descrição**: `MAPPOAgent.epsilon`, `get_epsilon()` e `decay_epsilon()` são computados mas **nunca usados** na seleção de ação (que usa `Categorical(probs).sample()` diretamente). `EPSILON_START`/`EPSILON_END`/`EPSILON_DECAY` no `MAPPOConfig` não impactam o comportamento.

**Nota**: Isto não é um bug — PPO on-policy usa entropia da distribuição categórica para exploração. O epsilon-greedy seria incompatível com o IS ratio. Mantido para compatibilidade de interface e logging.

---

### N13 — HATRPO: epsilon dead code (mesmo problema do MAPPO)

**Severidade**: 🟡 MÉDIO — **✅ DOCUMENTADO**  
**Arquivo**: `src/agents/hatrpo.py:49-59`

**Descrição**: Mesma situação do MAPPO: `epsilon` e `get_epsilon()` são computados mas ignorados por `select_action()`. A exploração vem da entropia da `Categorical`.

---

### N14 — `VALUE_LOSS_COEF` nunca usado no MAPPO

**Severidade**: 🟡 MÉDIO — **✅ CORRIGIDO** (Fase 9)  
**Arquivo**: `src/config.py:165`, `src/agents/mappo.py:159`

**Correção aplicada**: `critic_loss = self.config.VALUE_LOSS_COEF * F.mse_loss(batch_values, batch_returns)`

---

### N15 — `LEARNING_STARTS` ignorado em MAPPO, HATRPO e QMIX

**Severidade**: 🟢 BAIXO — 🔴 PENDENTE  
**Arquivos**: `src/config.py:173-174, 187`

**Descrição**: `MAPPOConfig.LEARNING_STARTS = 1000` e `HATRPOConfig.LEARNING_STARTS = 1000` estão definidos mas nunca verificados — `update()`/`update_actor()` são chamados desde o primeiro episódio. `QMIXTrainer.optimize()` também não verifica `steps_done` ou `learning_starts`.

**Correção esperada**: Adicionar verificação nos respectivos métodos `update()`/`optimize()` ou remover dos configs.

---

### N16 — QMIXTrainer não verifica `LEARNING_STARTS`

**Severidade**: 🟡 MÉDIO — **✅ CORRIGIDO** (Fase 9)  
**Arquivo**: `src/agents/qmix.py:117`

**Correção aplicada**: Adicionado `_total_steps` ao `QMIXTrainer`; `optimize()` agora retorna 0 se `_total_steps < LEARNING_STARTS`.

---

### N17 — VDN `_get_beta` superestima denominador do annealing

**Severidade**: 🟡 MÉDIO — 🔴 PENDENTE  
**Arquivo**: `src/agents/vdn.py:67-73`

**Descrição**: `_get_beta()` calcula fração como `steps_done / (EPISODES_TOTAL * MAX_STEPS)`. Isso pressupõe que todo episódio dura `MAX_STEPS` passos. Episódios bem-sucedidos terminam mais cedo, então o denominador é superestimado e o annealing do beta fica mais lento que o ideal.

**Correção esperada**: Usar `steps_done` real em vez de `EPISODES_TOTAL * MAX_STEPS`, ou rastrear steps reais.

---

### N18 — `distance_traveled` nunca plotado

**Severidade**: 🟢 BAIXO — **✅ CORRIGIDO** (Fase 9)  
**Arquivo**: `src/evaluation.py`

**Correção aplicada**: Subplot 6 substituído por gráfico de "Distância Percorrida" com fallback para painel de resumo se ausente.

---

## 📊 Matriz de Bugs por Algoritmo (Atualizada)

| Bug                                 | IDQN   | Random | VDN    | QMIX   | MAPPO  | HATRPO |
| ----------------------------------- | ------ | ------ | ------ | ------ | ------ | ------ |
| B1 (\_pickup_box)                   | ✅     | ✅     | ✅     | ✅     | ✅     | ✅     |
| B2 (\_robot_carrying init)          | ✅     | ✅     | ✅     | ✅     | ✅     | ✅     |
| B3 (PER alpha)                      | ✅     | N/A    | ✅     | ✅     | N/A    | N/A    |
| B4 (epsilon-greedy on-policy)       | N/A    | N/A    | N/A    | N/A    | ✅     | ✅     |
| **B5 (\_should_optimize bloqueia)** | **✅** | **✅** | N/A    | N/A    | N/A    | N/A    |
| **B6 (dropout ativo inferência)**   | **✅** | N/A    | **✅** | **✅** | ✅ OK  | **✅** |
| **B7 (HATRPO trust-region)**        | N/A    | N/A    | N/A    | N/A    | N/A    | **✅** |
| C4 (update_priorities alpha)        | ✅     | N/A    | N/A    | ✅     | N/A    | N/A    |
| C5 (\_max_priority double-alpha)    | N/A    | N/A    | ✅     | N/A    | N/A    | N/A    |
| I1 (residual critic)                | N/A    | N/A    | N/A    | N/A    | N/A    | ✅     |
| I2 (squeeze batch=1)                | ✅     | N/A    | N/A    | N/A    | N/A    | N/A    |
| I3 (variable rebind mappo)          | N/A    | N/A    | N/A    | N/A    | ✅     | N/A    |
| I5 (distance_traveled)              | ✅     | ✅     | N/A    | N/A    | N/A    | N/A    |
| I6 (checkpoint filename)            | ✅     | ✅     | N/A    | N/A    | N/A    | N/A    |
| I7 (IDQN dropout inferência)        | ✅     | N/A    | N/A    | N/A    | N/A    | N/A    |
| I8 (select_action threads)          | ✅     | ✅     | N/A    | N/A    | N/A    | N/A    |
| N1 (CLIP_EPS hardcoded)             | N/A    | N/A    | N/A    | N/A    | N/A    | ✅     |
| N2 (TAU hardcoded)                  | N/A    | N/A    | N/A    | N/A    | N/A    | ✅     |
| N8 (_foreach_lerp_)                 | ✅     | N/A    | ✅     | ✅     | N/A    | ✅     |
| **N9 (pool shutdown)**              | **✅** | **✅** | N/A    | N/A    | N/A    | N/A    |
| **N10 (bg_pool não aguardado)**     | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |
| **N11 (beta annealing quebrado)**   | **✅** | N/A    | N/A    | N/A    | N/A    | N/A    |
| N12/N13 (epsilon dead code)         | N/A    | N/A    | N/A    | N/A    | 📝     | 📝     |
| N14 (value_loss_coef)               | N/A    | N/A    | N/A    | N/A    | **✅** | N/A    |
| N15 (learning_starts ignorado)      | ✅     | N/A    | ✅     | 🔴     | 🔴     | 🔴     |
| N16 (QMIX sem learning_starts)      | N/A    | N/A    | N/A    | **✅** | N/A    | N/A    |
| N17 (VDN beta annealing lento)      | N/A    | N/A    | 🔴     | N/A    | N/A    | N/A    |
| N18 (distance_traveled não plotado) | **✅** | **✅** | **✅** | **✅** | **✅** | **✅** |

**Legenda**: ✅ = corrigido / 📝 = documentado, sem correção (intencional) / 🔴 = pendente / N/A = não se aplica

---

## 💡 Oportunidade de Unificação: `BaseAgent` e `BaseRunner`

A arquitetura ideal a longo prazo seria:

```
src/
├── agents/
│   ├── base_agent.py      # BaseAgent: select_action, remember, optimize interface
│   ├── base_runner.py     # BaseRunner: loop de treino, logging, CSV, plot, vídeo
│   ├── idqn.py            # IDQNAgent(BaseAgent)
│   ├── vdn.py             # VDNController(BaseAgent)
│   ├── qmix.py            # QMIXAgent(BaseAgent) + QMIXTrainer
│   ├── mappo.py           # MAPPOAgent(BaseAgent)
│   └── hatrpo.py          # HATRPOAgent(BaseAgent)
├── config.py              # EnvConfig + AlgoConfig (dataclasses)
├── environment.py         # WarehouseEnv
├── replay_buffer.py       # SumTree + PER buffers
├── networks.py            # Todas as redes
├── evaluation.py          # eval consolidado
└── main.py                # dispatch por --algo
```

Isso eliminaria:

- **D1**: `compute_gae` duplicado → método em `BaseAgent`
- **D2/D3**: logging/CSV/plot duplicados → métodos em `BaseRunner`
- **D4**: `soft_update_target` duplicado → método em `BaseAgent`
- **B5/B6/B7**: bugs comuns eliminados por herança
- **M1–M6**: configuração unificada via `@dataclass`
- **N7**: runner único
- **N8**: soft update vetorizado uma vez
- **E1**: GAE O(n²) corrigido na base
- **E2/E3**: otimizações na base
