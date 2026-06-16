"""Configurações e mapa do ambiente Warehouse.

Centraliza o ``MAP_CONFIG`` (antes duplicado em cada script/notebook) e as
classes de configuração. ``BaseConfig`` reúne os parâmetros usados pelo ambiente
e pelo loop de treino; ``IDQNConfig`` acrescenta os hiperparâmetros do IDQN e
``RandomConfig`` serve ao baseline aleatório.
"""

# Mapa da grade do armazém:
#   '0' livre | 'X' parede | 'Y' barreira | 'R1'/'R2' robôs
#   'A' caixa (origem) | 'B' alvo (entrega)
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

    MAX_STEPS = 500
    EPISODES_PER_SESSION = 1500
    BASE_DIR = "resultados_warehouse"
    SAVE_CHECKPOINTS = True
    SAVE_CHECKPOINT_EVERY = 500


class IDQNConfig(BaseConfig):
    """Hiperparâmetros do agente IDQN (porta de ``OptimizedConfig`` da v1.3.0)."""

    # Exploração
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 50000

    # Aprendizado
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 256
    GAMMA = 0.95
    TAU = 0.001
    WEIGHT_DECAY = 1e-5

    # Rede neural
    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.2

    # Memória / replay
    BUFFER_SIZE = 500000
    PRIORITIZED_REPLAY = True
    ALPHA = 0.6
    BETA = 0.4

    # Treinamento
    MAX_GRAD_NORM = 1.0
    LEARNING_STARTS = 1000
    TRAIN_FREQ = 4
    USE_SOFT_UPDATE = True

    BASE_DIR = "resultados_warehouse_idqn"


class RandomConfig(BaseConfig):
    """Configuração do baseline de ações aleatórias (sem aprendizado)."""

    BASE_DIR = "resultados_warehouse_random"
    SAVE_CHECKPOINTS = False


class VDNConfig(BaseConfig):
    """Value Decomposition Networks.

    Extraído de: Código/Executa Experimento VDN.ipynb (VDNConfig). Campos de
    falhas do ambiente foram omitidos (o env simples não os usa).
    """

    EPSILON_START = 1.0
    EPSILON_END = 0.02
    EPSILON_DECAY_STEPS = 120000

    LEARNING_RATE = 0.0003
    BATCH_SIZE = 256
    GAMMA = 0.97
    TAU = 0.005
    WEIGHT_DECAY = 1e-5

    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.1

    BUFFER_SIZE = 200000
    ALPHA = 0.6
    BETA_START = 0.4
    BETA_END = 1.0

    MAX_GRAD_NORM = 10.0
    LEARNING_STARTS = 2000
    TRAIN_FREQ = 2
    TARGET_UPDATE_FREQ = 200
    USE_SOFT_UPDATE = True

    MAX_STEPS = 1000
    EPISODES_TOTAL = 1500  # default; o runner ajusta para o total real de episódios
    BASE_DIR = "resultados_warehouse_vdn"


class QMIXConfig(BaseConfig):
    """QMIX (mixer monotônico sobre estado global).

    Extraído de: Código/Executa - Experimento.ipynb (classe Config do QMIX).
    """

    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY_STEPS = 50000

    LEARNING_RATE = 0.0001
    BATCH_SIZE = 128
    GAMMA = 0.95
    TAU = 0.001
    WEIGHT_DECAY = 1e-5

    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.2
    MIXER_HIDDEN_DIM = 128

    BUFFER_SIZE = 100000
    ALPHA = 0.6
    BETA = 0.4

    MAX_GRAD_NORM = 1.0
    LEARNING_STARTS = 1000
    TRAIN_FREQ = 4
    USE_SOFT_UPDATE = True

    BASE_DIR = "resultados_warehouse_qmix"


class MAPPOConfig(BaseConfig):
    """Multi-Agent PPO (atores por agente + crítico centralizado).

    Extraído de: Código/Executa Experimento MPPO.ipynb (classe Config, 3ª versão).
    """

    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    GAMMA = 0.99
    LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENTROPY_COEF = 0.01
    VALUE_LOSS_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    PPO_EPOCHS = 10
    BATCH_SIZE = 64
    MINI_BATCH_SIZE = 32

    EPSILON_START = 1.0
    EPSILON_END = 0.05
    EPSILON_DECAY = 0.995
    LEARNING_STARTS = 1000

    BASE_DIR = "resultados_warehouse_mappo"


class HATRPOConfig(BaseConfig):
    """HATRPO (atores trust-region sequenciais + crítico centralizado).

    Extraído de: Código/Executa Experimento HATRPO.ipynb (classe HATRPOConfigOptimized).
    """

    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_STEPS = 150000

    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-3
    GAMMA = 0.99
    LAMBDA = 0.95
    MAX_KL = 0.02

    HIDDEN_DIM = 512
    DROPOUT_RATE = 0.1
    NUM_LAYERS = 3

    MAX_GRAD_NORM = 1.0
    ENTROPY_COEF = 0.01
    USE_GRADIENT_CLIPPING = True
    TARGET_UPDATE_FREQ = 100

    BASE_DIR = "resultados_warehouse_hatrpo"
