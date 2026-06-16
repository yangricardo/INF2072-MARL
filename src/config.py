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
