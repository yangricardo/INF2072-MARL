"""Ponto de entrada por linha de comando.

Exemplos:
    python -m src.main --algo idqn --episodes 5
    python -m src.main --algo random --episodes 5 --no-video
    python -m src.main --algo idqn --sessions 2
"""

import argparse
import traceback

from .agents import IDQNAgent, RandomAgent
from .config import IDQNConfig, RandomConfig
from .training import run_training

ALGORITHMS = {
    "idqn": (IDQNAgent, IDQNConfig),
    "random": (RandomAgent, RandomConfig),
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
        description="Treino MARL no ambiente Warehouse (IDQN ou baseline aleatório)."
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

    agent_class, config_class = ALGORITHMS[args.algo]
    config = build_config(config_class, args)

    try:
        agents, metrics, video_path = run_training(
            agent_class,
            config,
            num_sessions=args.sessions,
            record_video=not args.no_video,
        )

        print("\n" + "=" * 60)
        print("✨ TREINAMENTO E AVALIAÇÃO CONCLUÍDOS COM SUCESSO! ✨")
        print("=" * 60)
        if video_path:
            print(f"\n📹 Vídeo gerado em: {video_path}")
    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
