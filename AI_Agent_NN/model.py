from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


@dataclass
class ModelConfig:
    board_size: int = 19
    in_planes: int = 6
    channels: int = 64
    num_res_blocks: int = 4
    policy_size: int = 19 * 19


@keras.utils.register_keras_serializable(package="Custom")
class ResidualBlock(layers.Layer):
    """
    Standard residual block:
        Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """

    def __init__(self, channels: int, name: str | None = None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.channels = channels

        self.conv1 = layers.Conv2D(
            filters=channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"{name}_conv1" if name else None,
        )
        self.bn1 = layers.BatchNormalization(name=f"{name}_bn1" if name else None)
        self.relu1 = layers.ReLU(name=f"{name}_relu1" if name else None)

        self.conv2 = layers.Conv2D(
            filters=channels,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name=f"{name}_conv2" if name else None,
        )
        self.bn2 = layers.BatchNormalization(name=f"{name}_bn2" if name else None)
        self.add = layers.Add(name=f"{name}_add" if name else None)
        self.relu2 = layers.ReLU(name=f"{name}_relu2" if name else None)

    def call(self, x, training: bool = False):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out, training=training)

        out = self.add([identity, out])
        out = self.relu2(out)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_policy_value_model(config: ModelConfig | None = None) -> keras.Model:
    """
    Builds a policy-value CNN for your 19x19 board game.

    Input shape:
        (board_size, board_size, in_planes)
        TensorFlow uses channels-last format.

    Outputs:
        policy_logits: shape (361,)
        value: shape (1,)
    """
    if config is None:
        config = ModelConfig()

    board_size = config.board_size
    in_planes = config.in_planes
    channels = config.channels
    num_res_blocks = config.num_res_blocks
    policy_size = config.policy_size

    if policy_size != board_size * board_size:
        raise ValueError(
            f"policy_size must equal board_size * board_size. "
            f"Got policy_size={policy_size}, board_size={board_size}"
        )

    inputs = keras.Input(
        shape=(board_size, board_size, in_planes),
        name="board_input",
    )

    # Stem
    x = layers.Conv2D(
        filters=channels,
        kernel_size=3,
        padding="same",
        use_bias=False,
        name="stem_conv",
    )(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    # Residual tower
    for i in range(num_res_blocks):
        x = ResidualBlock(channels=channels, name=f"resblock_{i+1}")(x)

    # Policy head
    p = layers.Conv2D(
        filters=2,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="policy_conv",
    )(x)
    p = layers.BatchNormalization(name="policy_bn")(p)
    p = layers.ReLU(name="policy_relu")(p)
    p = layers.Flatten(name="policy_flatten")(p)
    p = layers.Dense(
        units=policy_size,
        activation=None,
        name="policy_logits",
    )(p)

    # Value head
    v = layers.Conv2D(
        filters=1,
        kernel_size=1,
        padding="same",
        use_bias=False,
        name="value_conv",
    )(x)
    v = layers.BatchNormalization(name="value_bn")(v)
    v = layers.ReLU(name="value_relu")(v)
    v = layers.Flatten(name="value_flatten")(v)
    v = layers.Dense(128, activation="relu", name="value_dense_1")(v)
    v = layers.Dense(1, activation="tanh", name="value", dtype="float32")(v)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "policy_logits": p,
            "value": v,
        },
        name="go_policy_value_net",
    )

    return model


def get_default_model() -> keras.Model:
    """
    Convenience function for your exact first real model:
    - 19x19 board
    - 6 input planes
    - 64 channels
    - 4 residual blocks
    - 361 policy logits
    - 1 value scalar
    """
    config = ModelConfig(
        board_size=19,
        in_planes=6,
        channels=64,
        num_res_blocks=4,
        policy_size=19 * 19,
    )
    return build_policy_value_model(config)


def print_model_summary() -> None:
    model = get_default_model()
    model.summary()


if __name__ == "__main__":
    model = get_default_model()
    model.summary()

    dummy = tf.random.uniform(shape=(2, 19, 19, 6), dtype=tf.float32)
    outputs = model(dummy, training=False)

    print("\nSanity check:")
    print("policy_logits shape:", outputs["policy_logits"].shape)
    print("value shape:", outputs["value"].shape)