from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import tensorflow as tf

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_models import GoBoard, Stone
from AI_Agent_NN.model import ResidualBlock
from AI_Agent_NN.restricted_mcts import RestrictedMCTS


Move = Tuple[int, int]


class NeuralNetAI:
    """
    Runtime AI that:
    - loads best_model.keras
    - runs the policy-value network
    - uses restricted MCTS to choose the final move
    """

    def __init__(
        self,
        player: Stone,
        board_size: int = 19,
        model_path: str = "AI_Agent_NN/weights/best_model.keras",
        top_k: int = 5,
        simulations: int = 30,
        child_top_k: int = 3,
        c_puct: float = 1.5,
        max_depth: int = 10,
    ):
        self.player = player
        self.board_size = board_size
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Download best_model.keras and place it there first."
            )

        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={"ResidualBlock": ResidualBlock},
            compile=False,
        )

        self.search_engine = RestrictedMCTS(
            model=self.model,
            board_size=board_size,
            root_top_k=top_k,
            child_top_k=child_top_k,
            simulations=simulations,
            c_puct=c_puct,
            max_depth=max_depth,
        )

    def get_best_move(self, board: GoBoard) -> Optional[Move]:
        return self.search_engine.search(board)

    def get_debug_root_stats(self, board: GoBoard):
        return self.search_engine.debug_root_stats(board)


if __name__ == "__main__":
    board = GoBoard(size=19)
    ai = NeuralNetAI(player=Stone.BLACK)
    move = ai.get_best_move(board)
    print("Suggested move:", move)