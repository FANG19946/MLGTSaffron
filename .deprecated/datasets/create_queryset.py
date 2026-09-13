import numpy as np

X_PATH = "/mnt/Drive1/users/adnan/imagenet_work/X.npy"
Q_PATH = "/mnt/Drive1/users/adnan/imagenet_work/Q.npy"

rng = np.random.default_rng(42)

X = np.load(X_PATH, mmap_mode="r")

indices = rng.choice(X.shape[0], size=10_000, replace=False)
Q = X[indices]

np.save(Q_PATH, Q)

print(f"X shape: {X.shape}")
print(f"Q shape: {Q.shape}")
print(f"Saved Q to: {Q_PATH}")