"""Generate all datasets:  python -m scripts.make_data"""
import os
import time

import numpy as np

from src.data import generate

os.makedirs("data", exist_ok=True)
SPECS = {            # name: (episodes, seed, distribution)
    "train": (1000, 1, "train"),
    "val": (100, 2, "train"),
    "test": (100, 3, "train"),
    "test_ood": (100, 4, "ood"),
}
for name, (n, seed, dist) in SPECS.items():
    t0 = time.time()
    ds = generate(n, seed, dist)
    np.savez_compressed(f"data/{name}.npz", **ds)
    q, qd = ds["q"], ds["qd"]
    print(f"{name:9s} {n:5d} eps x {ds['u'].shape[1]} steps = {ds['u'].size:7d} samples | "
          f"accept {float(ds['acceptance']):.2f} | max|x| {np.abs(q[...,0]).max():.2f} | "
          f"max|qd| {np.abs(qd).max(axis=(0,1)).round(1)} | {time.time()-t0:.0f}s")
