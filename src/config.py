"""Configurações e mapa do ambiente Warehouse.

Centraliza o ``MAP_CONFIG`` (antes duplicado em cada script/notebook) e as
classes de configuração. ``BaseConfig`` reúne os parâmetros usados pelo ambiente
e pelo loop de treino; ``IDQNConfig`` acrescenta os hiperparâmetros do IDQN e
``RandomConfig`` serve ao baseline aleatório.

Dispositivo unificado (M5): ``DEVICE`` detecta automaticamente CUDA → MPS → CPU.
"""

# Mapa da grade do armazém:
#   '0' livre | 'X' parede | 'Y' barreira | 'R1'/'R2' robôs
#   'A' caixa (origem) | 'B' alvo (entrega)
import torch


def get_device():
    """Detecta o melhor device disponível: CUDA → Apple MPS → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


MAP_CONFIG = {
    "height": 12,
    "width": 8,
    "grid": [
        ["R1", "0", "0", "0", "0", "0", "0", "R2"],
        ["0", "Y", "A", "A", "A", "A", "0", "0"],
        ["0", "0", "A", "A", "A", "A", "0", "0"],
        ["X", "0", "0", "0", "X", "0", "Y", "0"],
        ["0", "Y", "X", "0", "0", "0", "0", "0"],
        ["0", "0", "0", "X", "0", "Y", "X", "X"],
        ["0", "X", "0", "Y", "0", "X", "0", "0"],
        ["0", "0", "0", "X", "0", "0", "0", "0"],
        ["X", "0", "Y", "0", "0", "0", "X", "0"],
        ["X", "0", "B", "B", "B", "B", "X", "0"],
        ["X", "0", "B", "B", "B", "B", "Y", "0"],
        ["0", "0", "0", "Y", "0", "0", "0", "0"],
    ],
}


class BaseConfig:
    """Parâmetros compartilhados pelo ambiente e pelo loop de treino."""

    # anotados explicitamente (não-Literal) pois são sobrescritos em runtime
    MAX_STEPS: int = 500
    EPISODES_PER_SESSION: int = 1500
    BASE_DIR: str = "resultados_warehouse"
    SAVE_CHECKPOINTS: bool = True
    SAVE_CHECKPOINT_EVERY: int = 500
    BATCH_SIZE: int = 256


class IDQNConfig(BaseConfig):
    """Hiperparâmetros do agente IDQN (porta de ``OptimizedConfig`` da v1.3.0). otimizados para o IDQN evitar o colapso de política e mínimos locais"""

    # 1. Exploração estendida (Garante que eles mapeiem o mapa antes de bitolar na rede)
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 200000  # Seu ajuste estendido

    # 2. Aprendizado Ágil (Combate a não-estacionaridade de múltiplos agentes)
    LEARNING_RATE = 0.00005  # LR mais baixo estabiliza o treino quando um interfere no outro
    BATCH_SIZE = 64          # Lotes menores focam em experiências temporalmente mais frescas
    GAMMA = 0.95
    TAU = 0.001
    WEIGHT_DECAY = 1e-5

    # Rede neural
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.2

    # 3. Correção do Replay Buffer Prioritário (Compensa o erro do contador de passos)
    BUFFER_SIZE = 500000
    PRIORITIZED_REPLAY = True
    ALPHA = 0.6
    BETA_START = 0.6         # Começa mais alto para mitigar o lento avanço do beta annealing
    BETA_END = 1.0

    # 4. Treinamento Agressivo 
    MAX_GRAD_NORM = 1.0
    LEARNING_STARTS = 1000
    TRAIN_FREQ = 1           # Otimiza a rede a cada 1 passo do ambiente (reação imediata)
    USE_SOFT_UPDATE = True

    BASE_DIR = "resultados_warehouse_idqn"


class RandomConfig(BaseConfig):
    """Configuração do baseline de ações aleatórias (sem aprendizado)."""

    BASE_DIR = "resultados_warehouse_random"
    SAVE_CHECKPOINTS = False


class VDNConfig(BaseConfig):
    """Value Decomposition Networks. Otimizado para Coordenação Linear

    Extraído de: Código/Executa Experimento VDN.ipynb (VDNConfig). Campos de
    falhas do ambiente foram omitidos (o env simples não os usa).
    """

    # Exploração robusta para o espaço de ações conjuntas
    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY_STEPS = 200000  # Aumentado para dar tempo de alinhar trajetórias

    # Aprendizado Fatorado Estabilizado
    LEARNING_RATE = 0.0001        # Reduzido de 0.0003 para evitar oscilações no Q_tot
    BATCH_SIZE = 64               # Ajustado para fatias temporais mais frescas
    GAMMA = 0.97
    TAU = 0.001                   # Mantém o soft update idêntico ao IDQN para consistência
    WEIGHT_DECAY = 1e-5

    HIDDEN_DIM = 512              # Sincronizado para 512 para parear capacidade de rede com o IDQN
    DROPOUT_RATE = 0.2            # Aumentado de 0.1 para 0.2 para evitar overfitting no mapa estático

    # Memória / Replay Prioritário Conjunto
    BUFFER_SIZE = 500000
    PRIORITIZED_REPLAY = True
    ALPHA = 0.6
    BETA_START = 0.6              # Ajustado para mitigar o erro do contador de passos do motor de treino
    BETA_END = 1.0

    # Atualização de Alvo Exclusiva (Removido o conflito com TARGET_UPDATE_FREQ)
    MAX_GRAD_NORM = 1.0           # Estabilizado em 1.0 (10.0 era muito permissivo)
    LEARNING_STARTS = 2000
    TRAIN_FREQ = 1                # Treino a cada passo para mitigar não-estacionaridade
    USE_SOFT_UPDATE = True

    MAX_STEPS = 500               # Sincronizado com a base (1000 quebrava o balanceamento de passos)
    BASE_DIR = "resultados_warehouse_vdn_otimizado"


class QMIXConfig(BaseConfig):
    """QMIX (mixer monotônico sobre estado global).

    Extraído de: Código/Executa - Experimento.ipynb (classe Config do QMIX).
    """

    # Exploração
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 200000  # Sincronizado para garantir mapeamento do mapa

    # Aprendizado com Hiper-redes Estabilizadas
    LEARNING_RATE = 0.00005       # LR menor é mandatório para conter os gradientes das hiper-redes
    BATCH_SIZE = 64               # Batches menores dão atualizações mais rápidas
    GAMMA = 0.95
    TAU = 0.001
    WEIGHT_DECAY = 1e-5

    # Arquitetura da Rede e do Mixer Monotônico
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.2
    MIXER_HIDDEN_DIM = 128

    # Memória / Replay Prioritário
    BUFFER_SIZE = 500000          # Expandido de 100k para 500k para armazenar trajetórias ricas
    ALPHA = 0.6
    BETA = 0.6                    # Ajustado para 0.6 para compensar o avanço de passos

    # Treinamento Dinâmico contra Variações de Estado Global
    MAX_GRAD_NORM = 1.0
    LEARNING_STARTS = 1000
    TRAIN_FREQ = 1                # Atualização imediata a cada passo
    USE_SOFT_UPDATE = True

    BASE_DIR = "resultados_warehouse_qmix_otimizado"


class MAPPOConfig(BaseConfig):
    """Multi-Agent PPO (atores por agente + crítico centralizado).

    Extraído de: Código/Executa Experimento MPPO.ipynb (classe Config, 3ª versão).
    """

    # Otimização de Atores e Crítico Centralizado (GAE)
    ACTOR_LR = 0.0003
    CRITIC_LR = 0.001             # Crítico precisa aprender mais rápido que o ator para estabilizar GAE
    GAMMA = 0.99
    LAMBDA = 0.95                 # Hiperparâmetro padrão para o cálculo de Vantagem GAE

    # PPO Clipping e Regularização Baseada em Entropia
    CLIP_EPS = 0.2                # Região de restrição da razão de probabilidade do PPO clip
    ENTROPY_COEF = 0.02           # Aumentado de 0.01 para 0.02 para forçar o Ator a manter o Softmax amplo e explorar
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 10               # Número de passadas de otimização sobre a mesma trajetória coletada

    # Tamanho do Buffer On-Policy por Episódio
    BATCH_SIZE = 64
    MINI_BATCH_SIZE = 32

    # REMOVIDOS os parâmetros de Epsilon-Greedy (Incompatíveis com amostragem estocástica via Categorical)
    # REMOVIDO LEARNING_STARTS para permitir que o algoritmo atualize logo após o primeiro episódio coletado

    BASE_DIR = "resultados_warehouse_mappo_otimizado"


class HATRPOConfig(BaseConfig):
    """HATRPO (atores trust-region sequenciais + crítico centralizado).
    — Otimizado para Região de Confiança Sequencial On-Policy.
    Extraído de: Código/Executa Experimento HATRPO.ipynb (classe HATRPOConfigOptimized).
    """

    # Parâmetros de exploração obsoletos por Epsilon foram eliminados da lógica de execução

    # Otimização das Redes Profundas Residuais
    ACTOR_LR = 0.0001              # Reduzido ligeiramente para garantir passos seguros na Região de Confiança
    CRITIC_LR = 0.001              # Reduzido de 3e-3 para 1e-3 para evitar explosão de gradiente no Crítico Centralizado
    GAMMA = 0.99
    LAMBDA = 0.95                  # GAE Lambda para balanceamento de Viés/Variância
    MAX_KL = 0.02
    CLIP_EPS = 0.2                 # Limite do substituto PPO para aproximar a restrição KL do TRPO tradicional

    # Configurações de Arquitetura das Redes (Improved Actor / Critic)
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.2             # Sincronizado em 0.2 para coibir memorização de caminhos vazios
    NUM_LAYERS = 3                 # Mantém as 3 camadas profundas para extração de features complexas

    # Regularização e Alvos do Crítico
    MAX_GRAD_NORM = 1.0
    ENTROPY_COEF = 0.02            # Força maior entropia na distribuição de probabilidades das ações iniciais
    USE_GRADIENT_CLIPPING = True
    TARGET_UPDATE_FREQ = 100       # Frequência de atualização da rede antiga do Ator (Mecanismo Trust-Region)
    TAU = 0.005                    # Taxa do Soft Update para a rede Target do Crítico Centralizado

    BASE_DIR = "resultados_warehouse_hatrpo_otimizado"