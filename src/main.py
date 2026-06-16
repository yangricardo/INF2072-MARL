"""Ponto de entrada por linha de comando.

Exemplos:
    python -m src.main --algo idqn --episodes 5
    python -m src.main --algo vdn --episodes 5 --no-video
    python -m src.main --algo random --episodes 5
"""

import argparse
import traceback

from .agents import IDQNAgent, RandomAgent, hatrpo, mappo, qmix, vdn
from .config import (
    HATRPOConfig,
    IDQNConfig,
    MAPPOConfig,
    QMIXConfig,
    RandomConfig,
    VDNConfig,
)
from .training import run_training


def _value_based_runner(agent_class):
    """Adapta o loop compartilhado (IDQN/Random) à assinatura uniforme de runner."""

    def runner(config, num_sessions, record_video):
        return run_training(
            agent_class, config, num_sessions=num_sessions, record_video=record_video
        )

    return runner


# algo -> (runner(config, num_sessions, record_video), config_class)
ALGORITHMS = {
    "idqn": (_value_based_runner(IDQNAgent), IDQNConfig),
    "random": (_value_based_runner(RandomAgent), RandomConfig),
    "vdn": (vdn.run, VDNConfig),
    "qmix": (qmix.run, QMIXConfig),
    "mappo": (mappo.run, MAPPOConfig),
    "hatrpo": (hatrpo.run, HATRPOConfig),
}


def build_config(config_class, args):
    config = config_class()
    if args.episodes is not None:
        config.EPISODES_PER_SESSION = args.episodes
    if args.output is not None:
        config.BASE_DIR = args.output
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Treino MARL no ambiente Warehouse (IDQN, Random, VDN, ...)."
    )
    parser.add_argument(
        "--algo",
        choices=sorted(ALGORITHMS.keys()),
        default="idqn",
        help="Algoritmo a treinar (padrão: idqn).",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=1,
        help="Número de sessões de treino (padrão: 1).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Sobrescreve EPISODES_PER_SESSION (útil para testes rápidos).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Diretório base de resultados (sobrescreve BASE_DIR).",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Não gravar o vídeo de avaliação ao final.",
    )
    args = parser.parse_args()

    runner, config_class = ALGORITHMS[args.algo]
    config = build_config(config_class, args)

    try:
        runner(config, num_sessions=args.sessions, record_video=not args.no_video)

        print("\n" + "=" * 60)
        print("✨ TREINAMENTO E AVALIAÇÃO CONCLUÍDOS COM SUCESSO! ✨")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
