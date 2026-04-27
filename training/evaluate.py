from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI_Agent_NN.model import ResidualBlock
from training.dataset_utils import BOARD_SIZE


def list_shards(dataset_root: str | Path, split: str) -> List[Path]:
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    shard_paths = sorted(split_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in {split_dir}")

    return shard_paths


def load_split_arrays(dataset_root: str | Path, split: str) -> Dict[str, np.ndarray]:
    shard_paths = list_shards(dataset_root, split)

    boards_list = []
    to_play_list = []
    last_moves_list = []
    move_indices_list = []
    winners_list = []

    for shard_path in shard_paths:
        with np.load(shard_path) as data:
            boards_list.append(data["boards"])
            to_play_list.append(data["to_play"])
            last_moves_list.append(data["last_moves"])
            move_indices_list.append(data["move_indices"])
            winners_list.append(data["winners"])

    return {
        "boards": np.concatenate(boards_list, axis=0).astype(np.int8),
        "to_play": np.concatenate(to_play_list, axis=0).astype(np.uint8),
        "last_moves": np.concatenate(last_moves_list, axis=0).astype(np.int16),
        "move_indices": np.concatenate(move_indices_list, axis=0).astype(np.int32),
        "winners": np.concatenate(winners_list, axis=0).astype(np.int8),
    }


def encode_example(board: np.ndarray, to_play: int, last_move: np.ndarray) -> np.ndarray:
    """
    Build the same 6 input planes used in training.
    Output shape: (19, 19, 6)
    """
    planes = np.zeros((BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)

    black = (board == 1)
    white = (board == 2)
    empty = (board == 0)

    if to_play == 1:
        planes[:, :, 0] = black.astype(np.float32)
        planes[:, :, 1] = white.astype(np.float32)
    else:
        planes[:, :, 0] = white.astype(np.float32)
        planes[:, :, 1] = black.astype(np.float32)

    planes[:, :, 2] = empty.astype(np.float32)

    r, c = int(last_move[0]), int(last_move[1])
    if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
        planes[r, c, 3] = 1.0

    planes[:, :, 4] = 1.0 if to_play == 1 else 0.0
    planes[:, :, 5] = 1.0

    return planes


def build_model_inputs(split_arrays: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boards = split_arrays["boards"]
    to_play = split_arrays["to_play"]
    last_moves = split_arrays["last_moves"]
    move_indices = split_arrays["move_indices"]
    winners = split_arrays["winners"]

    n = len(boards)
    x = np.zeros((n, BOARD_SIZE, BOARD_SIZE, 6), dtype=np.float32)

    for i in range(n):
        x[i] = encode_example(boards[i], int(to_play[i]), last_moves[i])

    y_policy = move_indices.astype(np.int32)

    # Value target from player-to-move perspective
    y_value = np.where(winners == to_play, 1.0, -1.0).astype(np.float32).reshape(-1, 1)

    return x, y_policy, y_value


def unpack_predictions(predictions):
    if isinstance(predictions, dict):
        policy_logits = predictions["policy_logits"]
        value = predictions["value"]
    else:
        policy_logits = predictions[0]
        value = predictions[1]
    return policy_logits, value


def top_k_accuracy(policy_logits: np.ndarray, y_policy: np.ndarray, k: int) -> float:
    topk = np.argpartition(policy_logits, -k, axis=1)[:, -k:]
    hits = [(y_policy[i] in topk[i]) for i in range(len(y_policy))]
    return float(np.mean(hits))


def evaluate_split(
    model,
    dataset_root: str | Path,
    split: str,
    batch_size: int = 256,
    top_k: int = 5,
) -> Dict:
    split_arrays = load_split_arrays(dataset_root, split)
    x, y_policy, y_value = build_model_inputs(split_arrays)

    predictions = model.predict(x, batch_size=batch_size, verbose=1)
    policy_logits, value_pred = unpack_predictions(predictions)

    value_pred = np.asarray(value_pred).reshape(-1, 1)

    policy_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    value_loss_fn = tf.keras.losses.MeanSquaredError()

    policy_loss = float(policy_loss_fn(y_policy, policy_logits).numpy())
    value_loss = float(value_loss_fn(y_value, value_pred).numpy())

    top1 = float(np.mean(np.argmax(policy_logits, axis=1) == y_policy))
    topk = top_k_accuracy(policy_logits, y_policy, top_k)

    result = {
        "split": split,
        "num_samples": int(len(y_policy)),
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "top1_accuracy": top1,
        "top{}_accuracy".format(top_k): topk,
    }
    return result


def print_result(result: Dict, top_k: int) -> None:
    print(f"\n=== {result['split'].upper()} RESULTS ===")
    print(f"samples: {result['num_samples']}")
    print(f"policy_loss: {result['policy_loss']:.6f}")
    print(f"value_loss: {result['value_loss']:.6f}")
    print(f"top1_accuracy: {result['top1_accuracy']:.6f}")
    print(f"top{top_k}_accuracy: {result[f'top{top_k}_accuracy']:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Offline evaluation for the trained policy-value model.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="dataset_1000",
        help="Path to dataset root",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="AI_Agent_NN/weights/best_model.keras",
        help="Path to saved Keras model",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="val",
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for prediction",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k accuracy to report",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional path to save results as JSON",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"ResidualBlock": ResidualBlock},
        compile=False,
    )

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    all_results = []
    for split in splits:
        result = evaluate_split(
            model=model,
            dataset_root=args.dataset_root,
            split=split,
            batch_size=args.batch_size,
            top_k=args.top_k,
        )
        print_result(result, args.top_k)
        all_results.append(result)

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()