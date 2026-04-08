from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset_utils import (
    BOARD_SIZE,
    SOURCE_ID_TO_NAME,
    index_to_move,
    load_shard,
)


def inspect_split(split_dir: Path) -> Dict:
    shard_paths = sorted(split_dir.glob("shard_*.npz"))

    if not shard_paths:
        return {
            "split": split_dir.name,
            "num_shards": 0,
            "num_positions": 0,
            "num_games": 0,
            "winner_counts": {},
            "source_counts": {},
            "invalid_move_indices": 0,
            "non_empty_target_cells": 0,
            "invalid_to_play": 0,
            "invalid_winners": 0,
            "board_value_errors": 0,
            "sample_preview": [],
        }

    total_positions = 0
    game_ids = set()
    winner_counter = Counter()
    source_counter = Counter()

    invalid_move_indices = 0
    non_empty_target_cells = 0
    invalid_to_play = 0
    invalid_winners = 0
    board_value_errors = 0

    sample_preview: List[Dict] = []

    for shard_path in shard_paths:
        shard = load_shard(shard_path)

        boards = shard["boards"]
        to_play = shard["to_play"]
        last_moves = shard["last_moves"]
        move_indices = shard["move_indices"]
        winners = shard["winners"]
        plys = shard["plys"]
        source_ids = shard["source_ids"]
        shard_game_ids = shard["game_ids"]

        n = boards.shape[0]
        total_positions += n
        game_ids.update(map(int, shard_game_ids.tolist()))

        for i in range(n):
            board = boards[i]
            move_index = int(move_indices[i])
            winner = int(winners[i])
            player = int(to_play[i])
            source_id = int(source_ids[i])

            winner_counter[winner] += 1
            source_counter[source_id] += 1

            if not np.all(np.isin(board, [0, 1, 2])):
                board_value_errors += 1

            if player not in (1, 2):
                invalid_to_play += 1

            if winner not in (1, 2):
                invalid_winners += 1

            if not (0 <= move_index < BOARD_SIZE * BOARD_SIZE):
                invalid_move_indices += 1
                continue

            row, col = index_to_move(move_index, BOARD_SIZE)

            if board[row, col] != 0:
                non_empty_target_cells += 1

            if len(sample_preview) < 5:
                sample_preview.append(
                    {
                        "game_id": int(shard_game_ids[i]),
                        "ply": int(plys[i]),
                        "to_play": player,
                        "winner": winner,
                        "source_id": source_id,
                        "source_name": SOURCE_ID_TO_NAME.get(source_id, f"unknown_{source_id}"),
                        "move_index": move_index,
                        "move_rc": [int(row), int(col)],
                        "last_move": [int(last_moves[i][0]), int(last_moves[i][1])],
                        "num_black": int(np.sum(board == 1)),
                        "num_white": int(np.sum(board == 2)),
                        "num_empty": int(np.sum(board == 0)),
                    }
                )

    summary = {
        "split": split_dir.name,
        "num_shards": len(shard_paths),
        "num_positions": total_positions,
        "num_games": len(game_ids),
        "winner_counts": {int(k): int(v) for k, v in sorted(winner_counter.items())},
        "source_counts": {
            SOURCE_ID_TO_NAME.get(int(k), f"unknown_{k}"): int(v)
            for k, v in sorted(source_counter.items())
        },
        "invalid_move_indices": invalid_move_indices,
        "non_empty_target_cells": non_empty_target_cells,
        "invalid_to_play": invalid_to_play,
        "invalid_winners": invalid_winners,
        "board_value_errors": board_value_errors,
        "sample_preview": sample_preview,
    }
    return summary


def inspect_dataset(dataset_root: str | Path = "dataset_1000") -> Dict:
    dataset_root = Path(dataset_root)

    report = {
        "dataset_root": str(dataset_root),
        "splits": {},
    }

    total_positions = 0
    total_games = 0

    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        summary = inspect_split(split_dir)
        report["splits"][split] = summary
        total_positions += summary["num_positions"]
        total_games += summary["num_games"]

    report["totals"] = {
        "num_positions": total_positions,
        "num_games_sum_of_splits": total_games,
    }

    return report


def print_report(report: Dict) -> None:
    print("\n=== DATASET INSPECTION REPORT ===")
    print(f"dataset_root: {report['dataset_root']}")
    print(f"total_positions: {report['totals']['num_positions']}")
    print(f"sum_of_split_game_counts: {report['totals']['num_games_sum_of_splits']}")

    for split_name, summary in report["splits"].items():
        print(f"\n--- {split_name.upper()} ---")
        print(f"num_shards: {summary['num_shards']}")
        print(f"num_positions: {summary['num_positions']}")
        print(f"num_games: {summary['num_games']}")
        print(f"winner_counts: {summary['winner_counts']}")
        print(f"source_counts: {summary['source_counts']}")
        print(f"invalid_move_indices: {summary['invalid_move_indices']}")
        print(f"non_empty_target_cells: {summary['non_empty_target_cells']}")
        print(f"invalid_to_play: {summary['invalid_to_play']}")
        print(f"invalid_winners: {summary['invalid_winners']}")
        print(f"board_value_errors: {summary['board_value_errors']}")

        if summary["sample_preview"]:
            print("sample_preview:")
            for item in summary["sample_preview"]:
                print(
                    f"  game_id={item['game_id']} ply={item['ply']} "
                    f"to_play={item['to_play']} winner={item['winner']} "
                    f"source={item['source_name']} move={item['move_rc']} "
                    f"black={item['num_black']} white={item['num_white']} empty={item['num_empty']}"
                )


def main():
    parser = argparse.ArgumentParser(description="Inspect generated Go dataset shards.")
    parser.add_argument("--dataset-root", type=str, default="dataset_1000", help="Path to dataset root")
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save full inspection report as JSON",
    )
    args = parser.parse_args()

    report = inspect_dataset(args.dataset_root)
    print_report(report)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\nSaved JSON report to: {out_path}")


if __name__ == "__main__":
    main()