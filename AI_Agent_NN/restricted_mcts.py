from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_models import GoBoard, Stone


Move = Tuple[int, int]


@dataclass
class SearchNode:
    board: GoBoard
    move: Optional[Move] = None
    prior: float = 1.0
    parent: Optional["SearchNode"] = None
    depth: int = 0
    children: Dict[Move, "SearchNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class RestrictedMCTS:
    """
    Tiny neural-guided search:
    - get NN policy logits
    - mask illegal moves
    - keep top-k legal root moves
    - run a very small PUCT-style search
    - use NN value head at leaves
    - choose final move by highest visit count
    """

    def __init__(
        self,
        model,
        board_size: int = 19,
        root_top_k: int = 5,
        child_top_k: int = 3,
        simulations: int = 30,
        c_puct: float = 1.5,
        max_depth: int = 10,
    ):
        self.model = model
        self.board_size = board_size
        self.root_top_k = root_top_k
        self.child_top_k = child_top_k
        self.simulations = simulations
        self.c_puct = c_puct
        self.max_depth = max_depth

    # -------------------------
    # Board helpers
    # -------------------------

    def _copy_board(self, board: GoBoard) -> GoBoard:
        new_board = GoBoard(size=board.size)

        for row in range(board.size):
            for col in range(board.size):
                new_board.board[row][col] = board.board[row][col]

        new_board.current_player = board.current_player
        new_board.move_count = board.move_count
        new_board.captured_stones = board.captured_stones.copy()

        if hasattr(board, "last_move"):
            new_board.last_move = getattr(board, "last_move", None)

        return new_board

    def _is_legal_move(self, board: GoBoard, row: int, col: int) -> bool:
        if not board.is_valid_position(row, col):
            return False
        if not board.is_empty(row, col):
            return False

        test_board = self._copy_board(board)
        return bool(test_board.place_stone(row, col))

    def _get_legal_moves(self, board: GoBoard) -> List[Move]:
        legal_moves: List[Move] = []

        for row in range(board.size):
            for col in range(board.size):
                if self._is_legal_move(board, row, col):
                    legal_moves.append((row, col))

        return legal_moves

    # -------------------------
    # Model input / output
    # -------------------------

    def _encode_board(self, board: GoBoard) -> np.ndarray:
        """
        Build the same 6 planes used in training, channels-last:
        shape = (19, 19, 6)

        Planes:
          0 current player's stones
          1 opponent stones
          2 empty cells
          3 last move marker
          4 current-player-is-black plane
          5 constant ones plane
        """
        planes = np.zeros((self.board_size, self.board_size, 6), dtype=np.float32)

        to_play = board.current_player
        current_is_black = (to_play == Stone.BLACK)

        for row in range(self.board_size):
            for col in range(self.board_size):
                cell = board.board[row][col]

                if cell == Stone.BLACK:
                    if current_is_black:
                        planes[row, col, 0] = 1.0
                    else:
                        planes[row, col, 1] = 1.0
                elif cell == Stone.WHITE:
                    if current_is_black:
                        planes[row, col, 1] = 1.0
                    else:
                        planes[row, col, 0] = 1.0
                else:
                    planes[row, col, 2] = 1.0

        last_move = getattr(board, "last_move", None)
        if last_move is not None:
            r, c = last_move
            if 0 <= r < self.board_size and 0 <= c < self.board_size:
                planes[r, c, 3] = 1.0

        planes[:, :, 4] = 1.0 if current_is_black else 0.0
        planes[:, :, 5] = 1.0

        return planes

    def _predict(self, board: GoBoard) -> Tuple[np.ndarray, float]:
        x = self._encode_board(board)
        x = np.expand_dims(x, axis=0)  # (1, 19, 19, 6)

        outputs = self.model(x, training=False)

        if isinstance(outputs, dict):
            policy_logits = outputs["policy_logits"].numpy()[0]
            value = float(outputs["value"].numpy()[0][0])
        else:
            policy_logits = outputs[0].numpy()[0]
            value = float(outputs[1].numpy()[0][0])

        return policy_logits, value

    # -------------------------
    # Search helpers
    # -------------------------

    def _move_to_index(self, move: Move) -> int:
        row, col = move
        return row * self.board_size + col

    def _terminal_value(self, board: GoBoard, winner: Stone) -> float:
        """
        Return value from the perspective of board.current_player.
        """
        return 1.0 if winner == board.current_player else -1.0

    def _select_top_moves_with_priors(
        self,
        policy_logits: np.ndarray,
        legal_moves: List[Move],
        top_k: Optional[int],
    ) -> List[Tuple[Move, float]]:
        if not legal_moves:
            return []

        scored_moves: List[Tuple[Move, float]] = []
        for move in legal_moves:
            idx = self._move_to_index(move)
            scored_moves.append((move, float(policy_logits[idx])))

        scored_moves.sort(key=lambda x: x[1], reverse=True)

        if top_k is None:
            selected = scored_moves
        else:
            selected = scored_moves[: min(top_k, len(scored_moves))]

        logits = np.array([score for _, score in selected], dtype=np.float64)
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        return [(move, float(prob)) for (move, _), prob in zip(selected, probs)]

    def _expand_node(self, node: SearchNode, policy_logits: np.ndarray, root: bool = False) -> None:
        winner = node.board.get_winner()
        if winner is not None:
            return

        legal_moves = self._get_legal_moves(node.board)
        if not legal_moves:
            return

        top_k = self.root_top_k if root else self.child_top_k
        move_priors = self._select_top_moves_with_priors(policy_logits, legal_moves, top_k)

        for move, prior in move_priors:
            if move in node.children:
                continue

            child_board = self._copy_board(node.board)
            ok = child_board.place_stone(move[0], move[1])
            if not ok:
                continue

            child = SearchNode(
                board=child_board,
                move=move,
                prior=prior,
                parent=node,
                depth=node.depth + 1,
            )
            node.children[move] = child

    def _select_child(self, node: SearchNode) -> SearchNode:
        best_child: Optional[SearchNode] = None
        best_score = -float("inf")

        parent_visits = max(1, node.visit_count)

        for child in node.children.values():
            # IMPORTANT:
            # child.q_value is from the CHILD player's perspective.
            # Parent is choosing which child to enter, so convert to PARENT perspective.
            q = -child.q_value

            u = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_child = child

        if best_child is None:
            raise RuntimeError("Failed to select child from a non-empty node.")
        return best_child

    def _backpropagate(self, path: List[SearchNode], leaf_value: float) -> None:
        """
        leaf_value is from the perspective of the leaf node's player-to-move.
        Flip sign while moving upward because turns alternate.
        """
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    # -------------------------
    # Public API
    # -------------------------

    def search(self, board: GoBoard) -> Optional[Move]:
        root_board = self._copy_board(board)
        root = SearchNode(board=root_board, move=None, prior=1.0, parent=None, depth=0)

        root_policy_logits, _ = self._predict(root.board)
        self._expand_node(root, root_policy_logits, root=True)

        if not root.children:
            return None

        for _ in range(self.simulations):
            node = root
            path = [root]

            # Selection
            while node.children:
                node = self._select_child(node)
                path.append(node)

                if node.board.get_winner() is not None:
                    break
                if node.depth >= self.max_depth:
                    break

            # Evaluation / expansion
            winner = node.board.get_winner()

            if winner is not None:
                leaf_value = self._terminal_value(node.board, winner)

            elif node.depth >= self.max_depth:
                _, leaf_value = self._predict(node.board)

            else:
                policy_logits, leaf_value = self._predict(node.board)
                self._expand_node(node, policy_logits, root=False)

            self._backpropagate(path, leaf_value)

        # Final move = highest visit count; tie-break in ROOT perspective
        best_child = max(
            root.children.values(),
            key=lambda child: (child.visit_count, -child.q_value),
        )
        return best_child.move

    def debug_root_stats(self, board: GoBoard) -> List[Dict]:
        root_board = self._copy_board(board)
        root = SearchNode(board=root_board, move=None, prior=1.0, parent=None, depth=0)

        root_policy_logits, _ = self._predict(root.board)
        self._expand_node(root, root_policy_logits, root=True)

        if not root.children:
            return []

        for _ in range(self.simulations):
            node = root
            path = [root]

            while node.children:
                node = self._select_child(node)
                path.append(node)

                if node.board.get_winner() is not None:
                    break
                if node.depth >= self.max_depth:
                    break

            winner = node.board.get_winner()

            if winner is not None:
                leaf_value = self._terminal_value(node.board, winner)
            elif node.depth >= self.max_depth:
                _, leaf_value = self._predict(node.board)
            else:
                policy_logits, leaf_value = self._predict(node.board)
                self._expand_node(node, policy_logits, root=False)

            self._backpropagate(path, leaf_value)

        stats = []
        for move, child in root.children.items():
            stats.append(
                {
                    "move": move,
                    "prior": child.prior,
                    "visits": child.visit_count,
                    "q_value_child_perspective": child.q_value,
                    "q_value_root_perspective": -child.q_value,
                }
            )

        stats.sort(key=lambda x: (-x["visits"], -x["q_value_root_perspective"]))
        return stats