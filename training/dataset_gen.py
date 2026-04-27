from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from game_models import GoBoard, Stone

from AI_Agent_Albi.montecarlo import MonteCarloAI as AlbiMonteCarlo
from AI_Agent_Ankon.montecarlo import MonteCarloAI as AnkonMonteCarlo

from training.dataset_utils import (
    BOARD_SIZE,
    SOURCE_NAME_TO_ID,
    board_to_int_matrix,
    ensure_dataset_dirs,
    get_last_move_array,
    is_legal_move,
    move_to_index,
    save_shard,
    summarize_split_dir,
)

# ------------------------------
# REAL DATASET SETTINGS
# ------------------------------
# Stronger than the tiny pilot, but still practical.
TEACHER_CONFIGS = {
    "albi_mcts": {"simulations": 50, "time_limit": 0.05},
    "ankon_mcts": {"simulations": 50, "time_limit": 0.05},
}


def create_teacher(source_name: str, player: Stone):
    cfg = TEACHER_CONFIGS[source_name]

    if source_name == "albi_mcts":
        return AlbiMonteCarlo(
            player=player,
            simulations=cfg["simulations"],
            time_limit=cfg["time_limit"],
        )

    if source_name == "ankon_mcts":
        return AnkonMonteCarlo(
            player=player,
            simulations=cfg["simulations"],
            time_limit=cfg["time_limit"],
        )

    raise ValueError(f"Unknown teacher source_name: {source_name}")


def choose_sources_for_game(py_rng: random.Random) -> Tuple[str, str]:
    """
    Force every game to be Albi MCTS vs Ankon MCTS, with random color assignment.
    """
    if py_rng.random() < 0.5:
        return "albi_mcts", "ankon_mcts"
    return "ankon_mcts", "albi_mcts"


def choose_split_plan(num_games: int, seed: int) -> List[str]:
    """
    Split by completed game, not by position.
    """
    if num_games < 3:
        raise ValueError("num_games must be at least 3 so train/val/test all exist.")

    train_games = max(1, int(num_games * 0.80))
    val_games = max(1, int(num_games * 0.10))
    test_games = num_games - train_games - val_games

    if test_games < 1:
        test_games = 1
        train_games = max(1, train_games - 1)

    plan = (["train"] * train_games) + (["val"] * val_games) + (["test"] * test_games)
    py_rng = random.Random(seed)
    py_rng.shuffle(plan)
    return plan


def make_position_sample(
    board: GoBoard,
    move: Tuple[int, int],
    game_id: int,
    ply: int,
    source_name: str,
) -> Dict:
    row, col = move

    return {
        "board": board_to_int_matrix(board),
        "to_play": 1 if board.current_player == Stone.BLACK else 2,
        "last_move": get_last_move_array(board),
        "move_index": move_to_index((row, col)),
        "winner": 0,  # fill after game ends
        "ply": ply,
        "source_id": SOURCE_NAME_TO_ID[source_name],
        "game_id": game_id,
    }


def assign_winner_to_game_samples(samples: List[Dict], winner: Stone) -> None:
    winner_int = 1 if winner == Stone.BLACK else 2
    for sample in samples:
        sample["winner"] = winner_int


def simulate_one_game(
    game_id: int,
    py_rng: random.Random,
    max_game_moves: int,
) -> Tuple[Optional[List[Dict]], Dict]:
    """
    Simulate one MCTS-vs-MCTS game.
    Returns (samples, info). If samples is None, discard the game.
    """
    board = GoBoard(size=BOARD_SIZE)

    black_source, white_source = choose_sources_for_game(py_rng)

    black_ai = create_teacher(black_source, Stone.BLACK)
    white_ai = create_teacher(white_source, Stone.WHITE)

    samples: List[Dict] = []
    consecutive_passes = 0

    info = {
        "black_source": black_source,
        "white_source": white_source,
        "num_positions": 0,
        "discard_reason": None,
    }

    for ply in range(max_game_moves):
        winner = board.get_winner()
        if winner is not None:
            assign_winner_to_game_samples(samples, winner)
            info["num_positions"] = len(samples)
            return samples, info

        current_is_black = board.current_player == Stone.BLACK
        source_name = black_source if current_is_black else white_source
        current_ai = black_ai if current_is_black else white_ai

        move = current_ai.get_best_move(board)

        # Handle pass / no move
        if move is None:
            if hasattr(board, "pass_turn"):
                board.pass_turn()
                consecutive_passes += 1
                if consecutive_passes >= 2:
                    info["discard_reason"] = "two_passes_no_winner"
                    return None, info
                continue

            info["discard_reason"] = "teacher_returned_none_no_pass"
            return None, info

        consecutive_passes = 0

        if not isinstance(move, (tuple, list)) or len(move) != 2:
            info["discard_reason"] = "teacher_returned_invalid_move_format"
            return None, info

        row, col = int(move[0]), int(move[1])

        # Validate only the chosen move
        if not is_legal_move(board, row, col):
            info["discard_reason"] = "teacher_returned_illegal_move"
            return None, info

        # Save position BEFORE applying the move
        samples.append(
            make_position_sample(
                board=board,
                move=(row, col),
                game_id=game_id,
                ply=ply,
                source_name=source_name,
            )
        )

        ok = board.place_stone(row, col)
        if not ok:
            info["discard_reason"] = "place_stone_failed_after_validation"
            return None, info

    info["discard_reason"] = "max_game_moves_reached"
    return None, info


def flush_split_buffer(
    split: str,
    buffer: List[Dict],
    split_dirs: Dict[str, Path],
    shard_indices: Dict[str, int],
    shard_size: int,
    force: bool = False,
) -> int:
    if not buffer:
        return 0

    if (not force) and (len(buffer) < shard_size):
        return 0

    if force:
        chunk = buffer[:]
        buffer.clear()
    else:
        chunk = buffer[:shard_size]
        del buffer[:shard_size]

    shard_path = split_dirs[split] / f"shard_{shard_indices[split]:03d}.npz"
    save_shard(chunk, shard_path)
    shard_indices[split] += 1
    return len(chunk)


def generate_dataset(
    num_games: int = 1000,
    dataset_root: str | Path = "dataset_1000",
    shard_size: int = 5000,
    max_game_moves: int = 80,
    seed: int = 42,
) -> Dict:
    """
    Generate the real MCTS-vs-MCTS dataset.
    """
    if num_games < 3:
        raise ValueError("num_games must be at least 3.")

    split_dirs = ensure_dataset_dirs(dataset_root)
    split_plan = choose_split_plan(num_games, seed=seed)

    py_rng = random.Random(seed)

    buffers: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    shard_indices: Dict[str, int] = {"train": 0, "val": 0, "test": 0}

    completed_games = 0
    attempted_games = 0
    positions_generated = 0

    discard_counter: Counter = Counter()
    split_game_counter: Counter = Counter()
    teacher_position_counter: Counter = Counter()
    teacher_game_pair_counter: Counter = Counter()

    start_time = time.time()

    while completed_games < num_games:
        attempted_games += 1
        game_id = completed_games

        samples, info = simulate_one_game(
            game_id=game_id,
            py_rng=py_rng,
            max_game_moves=max_game_moves,
        )

        if samples is None:
            discard_reason = info["discard_reason"] or "unknown_discard"
            discard_counter[discard_reason] += 1

            if attempted_games > num_games * 20:
                raise RuntimeError(
                    "Too many discarded games. Stopping to avoid infinite generation loop."
                )
            continue

        split = split_plan[completed_games]
        split_game_counter[split] += 1

        black_source = info["black_source"]
        white_source = info["white_source"]
        teacher_game_pair_counter[f"{black_source}__vs__{white_source}"] += 1

        for sample in samples:
            buffers[split].append(sample)
            teacher_position_counter[int(sample["source_id"])] += 1

        positions_generated += len(samples)
        completed_games += 1

        flush_split_buffer(
            split=split,
            buffer=buffers[split],
            split_dirs=split_dirs,
            shard_indices=shard_indices,
            shard_size=shard_size,
            force=False,
        )

        if completed_games % 20 == 0 or completed_games == num_games:
            elapsed = time.time() - start_time
            print(
                f"[progress] completed_games={completed_games}/{num_games}, "
                f"attempted_games={attempted_games}, "
                f"positions_generated={positions_generated}, "
                f"elapsed_sec={elapsed:.1f}"
            )

    for split in ("train", "val", "test"):
        flush_split_buffer(
            split=split,
            buffer=buffers[split],
            split_dirs=split_dirs,
            shard_indices=shard_indices,
            shard_size=shard_size,
            force=True,
        )

    metadata = {
        "description": "Real MCTS-vs-MCTS dataset generation (Albi vs Ankon every game).",
        "board_size": BOARD_SIZE,
        "num_requested_completed_games": num_games,
        "num_completed_games": completed_games,
        "num_attempted_games": attempted_games,
        "positions_generated": positions_generated,
        "seed": seed,
        "shard_size": shard_size,
        "max_game_moves": max_game_moves,
        "teacher_configs": TEACHER_CONFIGS,
        "source_name_to_id": SOURCE_NAME_TO_ID,
        "game_split_counts": dict(split_game_counter),
        "discard_counts": dict(discard_counter),
        "teacher_position_counts": {
            name: int(teacher_position_counter[source_id])
            for name, source_id in SOURCE_NAME_TO_ID.items()
            if name in TEACHER_CONFIGS
        },
        "teacher_game_pair_counts": dict(teacher_game_pair_counter),
        "split_summaries": {
            "train": summarize_split_dir(split_dirs["train"]),
            "val": summarize_split_dir(split_dirs["val"]),
            "test": summarize_split_dir(split_dirs["test"]),
        },
    }

    metadata_path = Path(dataset_root) / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("\nDataset generation complete.")
    print(f"Metadata written to: {metadata_path}")
    print("Split summaries:")
    print(json.dumps(metadata["split_summaries"], indent=2))

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Generate real MCTS-vs-MCTS Go dataset.")
    parser.add_argument("--num-games", type=int, default=1000, help="Number of completed games.")
    parser.add_argument("--dataset-root", type=str, default="dataset_1000", help="Output dataset directory.")
    parser.add_argument("--shard-size", type=int, default=5000, help="Positions per NPZ shard.")
    parser.add_argument("--max-game-moves", type=int, default=80, help="Max moves before discarding a game.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_dataset(
        num_games=args.num_games,
        dataset_root=args.dataset_root,
        shard_size=args.shard_size,
        max_game_moves=args.max_game_moves,
        seed=args.seed,
    )