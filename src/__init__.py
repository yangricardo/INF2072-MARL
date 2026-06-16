"""Pacote MARL Warehouse.

Reestruturação em módulos do código de Aprendizado por Reforço Multi-Agente
(originalmente espalhado em scripts e notebooks). Disponibiliza o ambiente
``WarehouseEnv`` e os agentes IDQN e Random num núcleo reutilizável.
"""

__all__ = [
    "config",
    "environment",
    "replay_buffer",
    "networks",
    "training",
    "evaluation",
    "agents",
]
