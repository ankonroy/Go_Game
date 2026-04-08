from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Make project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from AI_Agent_NN.model import ModelConfig, build_policy_value_model
from training.config import TrainConfig
from training.dataset_utils import apply_symmetry_2d, apply_symmetry_to_point


AUTOTUNE = tf.data.AUTOTUNE


def list_shards(dataset_root: str | Path, split: str) -> list[Path]:
    split_dir = Path(dataset_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    shard_paths = sorted(split_dir.glob("shard_*.npz"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard_*.npz files found in: {split_dir}")
    return shard_paths


def load_split_arrays(
    dataset_root: str | Path,
    split: str,
    limit_samples: int | None = None,
) -> Dict[str, np.ndarray]:
    shard_paths = list_shards(dataset_root, split)

    boards_list = []
    to_play_list = []
    last_moves_list = []
    move_indices_list = []
    winners_list = []

    total = 0
    for shard_path in shard_paths:
        with np.load(shard_path) as data:
            boards = data["boards"]
            to_play = data["to_play"]
            last_moves = data["last_moves"]
            move_indices = data["move_indices"]
            winners = data["winners"]

        boards_list.append(boards)
        to_play_list.append(to_play)
        last_moves_list.append(last_moves)
        move_indices_list.append(move_indices)
        winners_list.append(winners)

        total += len(boards)
        if limit_samples is not None and total >= limit_samples:
            break

    boards = np.concatenate(boards_list, axis=0)
    to_play = np.concatenate(to_play_list, axis=0)
    last_moves = np.concatenate(last_moves_list, axis=0)
    move_indices = np.concatenate(move_indices_list, axis=0)
    winners = np.concatenate(winners_list, axis=0)

    if limit_samples is not None:
        boards = boards[:limit_samples]
        to_play = to_play[:limit_samples]
        last_moves = last_moves[:limit_samples]
        move_indices = move_indices[:limit_samples]
        winners = winners[:limit_samples]

    return {
        "boards": boards.astype(np.int8),
        "to_play": to_play.astype(np.uint8),
        "last_moves": last_moves.astype(np.int16),
        "move_indices": move_indices.astype(np.int32),
        "winners": winners.astype(np.int8),
    }


def encode_planes_tf(
    board: tf.Tensor,
    to_play: tf.Tensor,
    last_move: tf.Tensor,
    board_size: int,
) -> tf.Tensor:
    """
    Build 6 input planes in channels-last format: (19, 19, 6)

    Planes:
      0 current player's stones
      1 opponent stones
      2 empty cells
      3 last move marker
      4 current-player-is-black plane
      5 constant ones plane
    """
    board = tf.cast(board, tf.int32)
    to_play = tf.cast(to_play, tf.int32)
    last_move = tf.cast(last_move, tf.int32)

    black = tf.equal(board, 1)
    white = tf.equal(board, 2)
    empty = tf.equal(board, 0)

    current_is_black = tf.equal(to_play, 1)

    my_stones = tf.where(current_is_black, black, white)
    opp_stones = tf.where(current_is_black, white, black)

    my_stones = tf.cast(my_stones, tf.float32)
    opp_stones = tf.cast(opp_stones, tf.float32)
    empty = tf.cast(empty, tf.float32)

    last_move_plane = tf.zeros((board_size, board_size), dtype=tf.float32)

    valid_last = tf.logical_and(
        tf.greater_equal(last_move[0], 0),
        tf.greater_equal(last_move[1], 0),
    )

    def _scatter_last() -> tf.Tensor:
        idx = tf.reshape(last_move, (1, 2))
        updates = tf.constant([1.0], dtype=tf.float32)
        return tf.tensor_scatter_nd_update(last_move_plane, idx, updates)

    last_move_plane = tf.cond(valid_last, _scatter_last, lambda: last_move_plane)

    current_player_plane = tf.where(
        current_is_black,
        tf.ones((board_size, board_size), dtype=tf.float32),
        tf.zeros((board_size, board_size), dtype=tf.float32),
    )

    ones_plane = tf.ones((board_size, board_size), dtype=tf.float32)

    planes = tf.stack(
        [
            my_stones,
            opp_stones,
            empty,
            last_move_plane,
            current_player_plane,
            ones_plane,
        ],
        axis=-1,
    )
    return planes


def winner_to_value_target_np(winners: np.ndarray, to_play: np.ndarray) -> np.ndarray:
    """
    Value target is from the perspective of the player to move.

      +1 if the eventual winner == player to move
      -1 otherwise
    """
    values = np.where(winners == to_play, 1.0, -1.0).astype(np.float32)
    return values.reshape(-1, 1)


def _augment_numpy(
    planes: np.ndarray,
    move_index: np.ndarray,
    board_size: int,
) -> tuple[np.ndarray, np.int32]:
    symmetry_id = np.random.randint(0, 8)

    aug_planes = np.stack(
        [apply_symmetry_2d(planes[:, :, c], symmetry_id) for c in range(planes.shape[-1])],
        axis=-1,
    ).astype(np.float32)

    row = int(move_index) // board_size
    col = int(move_index) % board_size
    new_row, new_col = apply_symmetry_to_point(row, col, symmetry_id, board_size)
    new_move_index = np.int32(new_row * board_size + new_col)

    return aug_planes, new_move_index


def maybe_augment_example(
    planes: tf.Tensor,
    policy_target: tf.Tensor,
    board_size: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    aug_planes, aug_policy = tf.numpy_function(
        func=lambda p, m: _augment_numpy(p, m, board_size),
        inp=[planes, policy_target],
        Tout=[tf.float32, tf.int32],
    )
    aug_planes.set_shape((board_size, board_size, 6))
    aug_policy.set_shape(())
    return aug_planes, aug_policy


def build_dataset(
    split_arrays: Dict[str, np.ndarray],
    cfg: TrainConfig,
    training: bool,
) -> tf.data.Dataset:
    boards = split_arrays["boards"]
    to_play = split_arrays["to_play"]
    last_moves = split_arrays["last_moves"]
    policy_targets = split_arrays["move_indices"]
    value_targets = winner_to_value_target_np(
        winners=split_arrays["winners"],
        to_play=split_arrays["to_play"],
    )

    ds = tf.data.Dataset.from_tensor_slices(
        (boards, to_play, last_moves, policy_targets, value_targets)
    )

    if training:
        ds = ds.shuffle(
            buffer_size=min(cfg.train_shuffle_buffer, len(boards)),
            seed=cfg.random_seed,
            reshuffle_each_iteration=True,
        )

    def _map_fn(board, turn, last_move, policy_target, value_target):
        planes = encode_planes_tf(
            board=board,
            to_play=turn,
            last_move=last_move,
            board_size=cfg.board_size,
        )

        if training and cfg.augment_symmetry:
            planes, policy_target = maybe_augment_example(
                planes=planes,
                policy_target=policy_target,
                board_size=cfg.board_size,
            )

        targets = {
            "policy_logits": tf.cast(policy_target, tf.int32),
            "value": tf.cast(value_target, tf.float32),
        }
        return planes, targets

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(cfg.batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_model_from_config(cfg: TrainConfig) -> keras.Model:
    model_cfg = ModelConfig(
        board_size=cfg.board_size,
        in_planes=cfg.in_planes,
        channels=cfg.channels,
        num_res_blocks=cfg.num_res_blocks,
        policy_size=cfg.num_policy_classes,
    )
    model = build_policy_value_model(model_cfg)
    return model


def compile_model(model: keras.Model, cfg: TrainConfig) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss={
            "policy_logits": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "value": keras.losses.MeanSquaredError(),
        },
        loss_weights={
            "policy_logits": cfg.policy_loss_weight,
            "value": cfg.value_loss_weight,
        },
        metrics={
            "policy_logits": [
                keras.metrics.SparseCategoricalAccuracy(name="policy_acc"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="policy_top5"),
            ],
            "value": [
                keras.metrics.MeanAbsoluteError(name="value_mae"),
            ],
        },
    )


def make_callbacks(cfg: TrainConfig) -> list[keras.callbacks.Callback]:
    cfg.ensure_dirs()

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.ModelCheckpoint(
            filepath=cfg.checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=False,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]
    return callbacks


def print_split_stats(name: str, split_arrays: Dict[str, np.ndarray]) -> None:
    n = len(split_arrays["boards"])
    winners = split_arrays["winners"]
    to_play = split_arrays["to_play"]

    black_wins = int(np.sum(winners == 1))
    white_wins = int(np.sum(winners == 2))
    black_to_play = int(np.sum(to_play == 1))
    white_to_play = int(np.sum(to_play == 2))

    print(f"\n{name} split:")
    print(f"  samples: {n}")
    print(f"  winner counts -> black: {black_wins}, white: {white_wins}")
    print(f"  to_play counts -> black: {black_to_play}, white: {white_to_play}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Go policy-value CNN with TensorFlow/Keras.")

    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root path")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Output .keras model path")

    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")

    parser.add_argument("--train-limit", type=int, default=None, help="Limit train samples for smoke test")
    parser.add_argument("--val-limit", type=int, default=None, help="Limit val samples for smoke test")

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Convenience mode: tiny subset + 1 epoch",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable random symmetry augmentation",
    )

    return parser.parse_args()


def apply_overrides(cfg: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    if args.dataset_root is not None:
        cfg.dataset_root = args.dataset_root
    if args.checkpoint_path is not None:
        cfg.checkpoint_path = args.checkpoint_path
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.no_augment:
        cfg.augment_symmetry = False

    if args.smoke_test:
        cfg.epochs = 1
        cfg.batch_size = min(cfg.batch_size, 128)

    return cfg


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    cfg = apply_overrides(cfg, args)

    tf.keras.utils.set_random_seed(cfg.random_seed)

    train_limit = args.train_limit
    val_limit = args.val_limit

    if args.smoke_test:
        if train_limit is None:
            train_limit = 12000
        if val_limit is None:
            val_limit = 2000

    print("\nLoading dataset...")
    print(f"dataset_root: {cfg.dataset_root}")
    print(f"checkpoint_path: {cfg.checkpoint_path}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"learning_rate: {cfg.learning_rate}")
    print(f"epochs: {cfg.epochs}")
    print(f"augment_symmetry: {cfg.augment_symmetry}")

    train_arrays = load_split_arrays(cfg.dataset_root, "train", limit_samples=train_limit)
    val_arrays = load_split_arrays(cfg.dataset_root, "val", limit_samples=val_limit)

    print_split_stats("Train", train_arrays)
    print_split_stats("Val", val_arrays)

    train_ds = build_dataset(train_arrays, cfg, training=True)
    val_ds = build_dataset(val_arrays, cfg, training=False)

    model = build_model_from_config(cfg)
    compile_model(model, cfg)

    print("\nModel summary:")
    model.summary()

    callbacks = make_callbacks(cfg)

    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nTraining finished.")
    print(f"Best checkpoint should be at: {cfg.checkpoint_path}")

    # Save final model too, even if checkpoint already saved the best one
    final_path = str(Path(cfg.checkpoint_path).with_name("last_model.keras"))
    model.save(final_path)
    print(f"Final model also saved to: {final_path}")


if __name__ == "__main__":
    main()