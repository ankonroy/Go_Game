from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    # Paths
    dataset_root: str = os.environ.get("GO_DATASET_ROOT", "dataset_1000")
    checkpoint_path: str = os.environ.get(
        "GO_CHECKPOINT_PATH", "AI_Agent_NN/weights/best_model.keras"
    )

    # Board / model
    board_size: int = 19
    in_planes: int = 6
    channels: int = 64
    num_res_blocks: int = 4
    num_policy_classes: int = 19 * 19

    # Training
    batch_size: int = 128
    learning_rate: float = 1e-3
    epochs: int = 12
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    train_shuffle_buffer: int = 20000
    random_seed: int = 42
    augment_symmetry: bool = True

    # Later inference/search settings
    top_k_moves: int = 5
    restricted_search_simulations: int = 30

    def checkpoint_dir(self) -> Path:
        return Path(self.checkpoint_path).resolve().parent

    def ensure_dirs(self) -> None:
        self.checkpoint_dir().mkdir(parents=True, exist_ok=True)