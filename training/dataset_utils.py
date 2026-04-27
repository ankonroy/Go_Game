from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

# Make project root importable when this file is imported indirectly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_models import GoBoard, Stone


BOARD_SIZE = 19
LAST_MOVE_NONE = (-1, -1)
SPLITS = ("train", "val", "test")

SOURCE_NAME_TO_ID: Dict[str, int] = {
    "albi_minimax": 0,
    "albi_mcts": 1,
    "ankon_minimax": 2,
    "ankon_mcts": 3,
}
SOURCE_ID_TO_NAME: Dict[int, str] = {v: k for k, v in SOURCE_NAME_TO_ID.items()}


def ensure_dataset_dirs(dataset_root: str | Path = "dataset") -> Dict[str, Path]:
    root = Path(dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    split_dirs: Dict[str, Path] = {}
    for split in SPLITS:
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split] = split_dir
    return split_dirs


def move_to_index(move: Tuple[int, int], board_size: int = BOARD_SIZE) -> int:
    row, col = move
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Move {move} is out of range for board size {board_size}.")
    return row * board_size + col


def index_to_move(index: int, board_size: int = BOARD_SIZE) -> Tuple[int, int]:
    if not (0 <= index < board_size * board_size):
        raise ValueError(f"Move index {index} is out of range.")
    return index // board_size, index % board_size


def normalize_last_move(last_move: Sequence[int] | Tuple[int, int] | None) -> np.ndarray:
    """
    Convert last_move into int16 array of shape (2,).
    None becomes [-1, -1].
    """
    if last_move is None:
        return np.array(LAST_MOVE_NONE, dtype=np.int16)

    if len(last_move) != 2:
        raise ValueError("last_move must have exactly 2 values.")

    row, col = int(last_move[0]), int(last_move[1])

    if (row, col) == LAST_MOVE_NONE:
        return np.array(LAST_MOVE_NONE, dtype=np.int16)

    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Invalid last_move: {(row, col)}")

    return np.array([row, col], dtype=np.int16)


def get_last_move_array(board: GoBoard) -> np.ndarray:
    last_move = getattr(board, "last_move", None)
    return normalize_last_move(last_move)


def board_to_int_matrix(board: GoBoard, board_size: int = BOARD_SIZE) -> np.ndarray:
    """
    Convert GoBoard.board to raw int matrix:
      0 = empty
      1 = black
      2 = white
    """
    arr = np.zeros((board_size, board_size), dtype=np.int8)

    for row in range(board_size):
        for col in range(board_size):
            cell = board.board[row][col]
            if cell == Stone.BLACK:
                arr[row, col] = 1
            elif cell == Stone.WHITE:
                arr[row, col] = 2
            else:
                arr[row, col] = 0

    return arr


def copy_board(board: GoBoard) -> GoBoard:
    """
    Copy GoBoard in the same style as your AI agents.
    """
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


def is_legal_move(board: GoBoard, row: int, col: int) -> bool:
    """
    Real legality check for one move only.
    This is much cheaper than computing a full legal mask for every position.
    """
    if not board.is_valid_position(row, col):
        return False
    if not board.is_empty(row, col):
        return False

    test_board = copy_board(board)
    return bool(test_board.place_stone(row, col))


def validate_board(board: np.ndarray, board_size: int = BOARD_SIZE) -> np.ndarray:
    arr = np.asarray(board, dtype=np.int8)

    if arr.shape != (board_size, board_size):
        raise ValueError(f"Board shape must be {(board_size, board_size)}, got {arr.shape}")

    if not np.all(np.isin(arr, [0, 1, 2])):
        raise ValueError("Board contains values outside {0,1,2}.")

    return arr


def validate_sample(sample: Mapping, board_size: int = BOARD_SIZE) -> Dict[str, np.ndarray | int]:
    required_keys = {
        "board",
        "to_play",
        "last_move",
        "move_index",
        "winner",
        "ply",
        "source_id",
        "game_id",
    }

    missing = required_keys - set(sample.keys())
    if missing:
        raise KeyError(f"Sample is missing required keys: {sorted(missing)}")

    board = validate_board(sample["board"], board_size)
    to_play = int(sample["to_play"])
    last_move = normalize_last_move(sample["last_move"])
    move_index = int(sample["move_index"])
    winner = int(sample["winner"])
    ply = int(sample["ply"])
    source_id = int(sample["source_id"])
    game_id = int(sample["game_id"])

    if to_play not in (1, 2):
        raise ValueError(f"to_play must be 1 or 2, got {to_play}")

    if winner not in (1, 2):
        raise ValueError(f"winner must be 1 or 2, got {winner}")

    if source_id not in SOURCE_ID_TO_NAME:
        raise ValueError(f"source_id must be one of {sorted(SOURCE_ID_TO_NAME.keys())}, got {source_id}")

    if ply < 0:
        raise ValueError(f"ply must be >= 0, got {ply}")

    if game_id < 0:
        raise ValueError(f"game_id must be >= 0, got {game_id}")

    row, col = index_to_move(move_index, board_size)

    # Since this is a "before move" board snapshot, chosen move must point to empty cell
    if board[row, col] != 0:
        raise ValueError(
            f"move_index={move_index} -> {(row, col)} points to a non-empty board cell."
        )

    return {
        "board": board,
        "to_play": to_play,
        "last_move": last_move,
        "move_index": move_index,
        "winner": winner,
        "ply": ply,
        "source_id": source_id,
        "game_id": game_id,
    }


def pack_samples(samples: Sequence[Mapping], board_size: int = BOARD_SIZE) -> Dict[str, np.ndarray]:
    if len(samples) == 0:
        raise ValueError("Cannot pack an empty sample list.")

    normalized = [validate_sample(s, board_size=board_size) for s in samples]

    shard = {
        "boards": np.stack([s["board"] for s in normalized]).astype(np.int8),
        "to_play": np.array([s["to_play"] for s in normalized], dtype=np.uint8),
        "last_moves": np.stack([s["last_move"] for s in normalized]).astype(np.int16),
        "move_indices": np.array([s["move_index"] for s in normalized], dtype=np.int16),
        "winners": np.array([s["winner"] for s in normalized], dtype=np.int8),
        "plys": np.array([s["ply"] for s in normalized], dtype=np.int16),
        "source_ids": np.array([s["source_id"] for s in normalized], dtype=np.uint8),
        "game_ids": np.array([s["game_id"] for s in normalized], dtype=np.int32),
    }

    validate_shard(shard, board_size=board_size)
    return shard


def validate_shard(shard: Mapping[str, np.ndarray], board_size: int = BOARD_SIZE) -> None:
    required_keys = {
        "boards",
        "to_play",
        "last_moves",
        "move_indices",
        "winners",
        "plys",
        "source_ids",
        "game_ids",
    }

    missing = required_keys - set(shard.keys())
    if missing:
        raise KeyError(f"Shard is missing required keys: {sorted(missing)}")

    boards = np.asarray(shard["boards"])
    to_play = np.asarray(shard["to_play"])
    last_moves = np.asarray(shard["last_moves"])
    move_indices = np.asarray(shard["move_indices"])
    winners = np.asarray(shard["winners"])
    plys = np.asarray(shard["plys"])
    source_ids = np.asarray(shard["source_ids"])
    game_ids = np.asarray(shard["game_ids"])

    n = boards.shape[0]

    if boards.shape != (n, board_size, board_size):
        raise ValueError(f"boards shape invalid: {boards.shape}")

    if to_play.shape != (n,):
        raise ValueError(f"to_play shape invalid: {to_play.shape}")

    if last_moves.shape != (n, 2):
        raise ValueError(f"last_moves shape invalid: {last_moves.shape}")

    if move_indices.shape != (n,):
        raise ValueError(f"move_indices shape invalid: {move_indices.shape}")

    if winners.shape != (n,):
        raise ValueError(f"winners shape invalid: {winners.shape}")

    if plys.shape != (n,):
        raise ValueError(f"plys shape invalid: {plys.shape}")

    if source_ids.shape != (n,):
        raise ValueError(f"source_ids shape invalid: {source_ids.shape}")

    if game_ids.shape != (n,):
        raise ValueError(f"game_ids shape invalid: {game_ids.shape}")


def save_shard(samples: Sequence[Mapping], output_path: str | Path, board_size: int = BOARD_SIZE) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shard = pack_samples(samples, board_size=board_size)
    np.savez_compressed(output_path, **shard)
    return output_path


def load_shard(shard_path: str | Path) -> Dict[str, np.ndarray]:
    shard_path = Path(shard_path)
    with np.load(shard_path) as data:
        shard = {key: data[key] for key in data.files}

    validate_shard(shard, board_size=BOARD_SIZE)
    return shard


def shard_summary(shard: Mapping[str, np.ndarray]) -> Dict[str, object]:
    boards = np.asarray(shard["boards"])
    winners = np.asarray(shard["winners"])
    source_ids = np.asarray(shard["source_ids"])
    game_ids = np.asarray(shard["game_ids"])

    unique_games = np.unique(game_ids)
    unique_sources, source_counts = np.unique(source_ids, return_counts=True)
    unique_winners, winner_counts = np.unique(winners, return_counts=True)

    return {
        "num_positions": int(boards.shape[0]),
        "num_games": int(len(unique_games)),
        "source_counts": {
            SOURCE_ID_TO_NAME[int(k)]: int(v)
            for k, v in zip(unique_sources, source_counts)
        },
        "winner_counts": {
            int(k): int(v)
            for k, v in zip(unique_winners, winner_counts)
        },
    }


def summarize_split_dir(split_dir: str | Path) -> Dict[str, object]:
    split_dir = Path(split_dir)
    shard_paths = sorted(split_dir.glob("shard_*.npz"))

    total_positions = 0
    total_games = set()
    source_counter: Counter = Counter()
    winner_counter: Counter = Counter()

    for shard_path in shard_paths:
        shard = load_shard(shard_path)
        total_positions += int(shard["boards"].shape[0])
        total_games.update(map(int, shard["game_ids"].tolist()))
        source_counter.update(map(int, shard["source_ids"].tolist()))
        winner_counter.update(map(int, shard["winners"].tolist()))

    return {
        "num_shards": len(shard_paths),
        "num_positions": total_positions,
        "num_games": len(total_games),
        "source_counts": {
            SOURCE_ID_TO_NAME[k]: int(v)
            for k, v in sorted(source_counter.items())
        },
        "winner_counts": {
            int(k): int(v)
            for k, v in sorted(winner_counter.items())
        },
    }


# Optional symmetry helpers for training later

def apply_symmetry_2d(arr: np.ndarray, symmetry_id: int) -> np.ndarray:
    """
    8 board symmetries.
    """
    if symmetry_id not in range(8):
        raise ValueError("symmetry_id must be in [0, 7]")

    out = np.asarray(arr)

    if symmetry_id == 0:
        return out.copy()
    if symmetry_id == 1:
        return np.rot90(out, 1).copy()
    if symmetry_id == 2:
        return np.rot90(out, 2).copy()
    if symmetry_id == 3:
        return np.rot90(out, 3).copy()
    if symmetry_id == 4:
        return np.fliplr(out).copy()
    if symmetry_id == 5:
        return np.rot90(np.fliplr(out), 1).copy()
    if symmetry_id == 6:
        return np.rot90(np.fliplr(out), 2).copy()
    return np.rot90(np.fliplr(out), 3).copy()


def apply_symmetry_to_point(row: int, col: int, symmetry_id: int, board_size: int = BOARD_SIZE) -> Tuple[int, int]:
    n = board_size
    r, c = row, col

    if symmetry_id == 0:
        return r, c
    if symmetry_id == 1:
        return c, n - 1 - r
    if symmetry_id == 2:
        return n - 1 - r, n - 1 - c
    if symmetry_id == 3:
        return n - 1 - c, r
    if symmetry_id == 4:
        return r, n - 1 - c
    if symmetry_id == 5:
        r, c = r, n - 1 - c
        return c, n - 1 - r
    if symmetry_id == 6:
        r, c = r, n - 1 - c
        return n - 1 - r, n - 1 - c
    if symmetry_id == 7:
        r, c = r, n - 1 - c
        return n - 1 - c, r

    raise ValueError("symmetry_id must be in [0, 7]")