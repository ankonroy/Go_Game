import numpy as np

data = np.load("dataset/train/shard_000.npz")
print(data.files)
print(data["boards"].shape)
print(data["move_indices"].shape)