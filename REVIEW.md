# Revisão de Código — LocalMultiAgente/src/

Análise completa do código em `src/` e subdiretórios, identificando bugs, inconsistências, código duplicado e oportunidades de otimização.

**Data da revisão**: 2026-06-15  
**Escopo**: src/ (agents/, config.py, environment.py, evaluation.py, main.py, networks.py, replay_buffer.py, training.py)

---

## ✅ Correções Aplicadas

| ID | Arquivo | Status | Descrição |
|---|---|---|---|
| C1 | src/networks.py | ✅ CORRIGIDO | QMixer: adicionado `torch.abs()` aos pesos W1 e W2 para garantir monotonicidade |
| C2 | src/environment.py | ✅ CORRIGIDO | `_drop_box`: adicionado dict `robot_carrying` para rastrear qual robô carrega qual caixa |
| C3 | src/environment.py | ✅ CORRIGIDO | Bônus terminal +50: movido para fora de `_calculate_shaped_reward`, aplicado uma única vez por step |
| C4 | src/replay_buffer.py | ✅ CORRIGIDO | `PrioritizedReplayBuffer` e `QMIXPrioritizedReplayBuffer`: adicionado `** self.alpha` em `update_priorities` |
| C5 | src/replay_buffer.py | ✅ CORRIGIDO | `VDNPrioritizedReplayBuffer`: `_max_priority` armazenado como valor raw, alpha aplicado apenas no armazenamento |
| I1 | src/networks.py | ✅ CORRIGIDO | `ImprovedCriticNetwork`: adicionada conexão residual no loop de camadas ocultas |
| I2 | src/agents/idqn.py | ✅ CORRIGIDO | `squeeze()` → `squeeze(1)` em 2 linhas para evitar colapso com batch_size=1 |
| I3 | src/agents/mappo.py | ✅ CORRIGIDO | Variável `values` rebind: renomeada para `batch_values` no loop de mini-batches |
| I5 | src/training.py | ✅ CORRIGIDO | `distance_traveled` adicionado à `consolidated_metrics` |
| I6 | src/training.py | ✅ CORRIGIDO | Checkpoint: nome alterado de `best_agent_{i}_ep{episode}.pth` para `best_agent_{i}_best.pth` |
| N1 | src/config.py, src/agents/hatrpo.py | ✅ CORRIGIDO | `CLIP_EPS`: adicionado ao `HATRPOConfig` e usado em lugar do hardcoded `0.8, 1.2` |
| N2 | src/config.py, src/agents/hatrpo.py | ✅ CORRIGIDO | `TAU`: adicionado ao `HATRPOConfig` e usado em lugar do hardcoded `0.005` |

---

## Pendente (não crítico)

| ID | Arquivo | Status | Descrição |
|---|---|---|---|
| I4 | src/agents/mappo.py, hatrpo.py | TODO | Usar retorno direto de `env.reset()` em vez de API privada |
| D1 | src/utils.py (novo) | TODO | Extrair `compute_gae` para função reutilizável |
| D2 | src/evaluation.py | TODO | Extrair consolidação de métricas para função reutilizável |
| E1 | src/agents/mappo.py, hatrpo.py | TODO | Mudar `advantages.insert(0)` para `append` + `reverse` (O(n) em vez de O(n²)) |
| E2 | src/agents/qmix.py | TODO | Reutilizar `curr_qs` em vez de chamar `get_q_values` duas vezes |

---

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

| Prioridade | ID | Arquivo(s) | Ação |
|---|---|---|---|
| CRÍTICO | C1 | networks.py | Adicionar `torch.abs()` em QMixer |
| CRÍTICO | C2 | environment.py | Rastrear `robot_carrying` |
| CRÍTICO | C3 | environment.py | Bônus terminal uma única vez |
| CRÍTICO | C4 | replay_buffer.py | Alpha em `update_priorities` (2 classes) |
| CRÍTICO | C5 | replay_buffer.py | `_max_priority` raw em VDN |
| ALTO | I1 | networks.py | Residual em `ImprovedCriticNetwork` |
| ALTO | I2 | idqn.py | `squeeze(1)` em 2 linhas |
| ALTO | I3 | mappo.py | Renomear variável interna |
| ALTO | I4 | mappo.py, hatrpo.py | Usar retorno de `env.reset()` |
| ALTO | I5 | training.py | Incluir `distance_traveled` |
| MÉDIO | I6 | training.py | Nome fixo para checkpoint |
| MÉDIO | N1 | hatrpo.py, config.py | CLIP_EPS configurável |
| MÉDIO | N2 | hatrpo.py, config.py | TAU configurável |
| BAIXO | D1–D4 | múltiplos | Extrair código duplicado |
| BAIXO | E1–E3 | mappo.py, hatrpo.py, qmix.py, env.py | Otimizações menores |

---

## 📚 Referências

- Rashid, T., et al. (2018). *QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning*. ICML 2018.
- Sunehag, P., et al. (2017). *Value-Decomposition Networks For Cooperative Multi-Agent Learning*. AAMAS 2018.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. OpenAI Technical Report.
