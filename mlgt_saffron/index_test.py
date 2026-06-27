#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
from argparse import ArgumentParser

from data_dir import DATASETS_DIR

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from mlgt_saffron import (
    OrionIndex,
    BloomHashFunction,
    computePools,
    normalizeDataset,
)

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]


if __name__ == "__main__":
    parser = ArgumentParser("Build Orion index")

    parser.add_argument("-p", "--data-path", type=str, default=DATASETS_DIR)
    parser.add_argument("-d", "--dataset", type=str, default="imagenet")
    parser.add_argument("-n", "--num-features", type=int, default=-1)
    parser.add_argument("-k", "--num-neighbors", type=int, default=10)

    parser.add_argument("--num-hashes", type=int, default=100)
    parser.add_argument("--hash-bits", type=int, default=16)
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("-v", "--verbose", type=int, default=0)

    args = parser.parse_args()

    print("Loading dataset...")

    data_path = os.path.join(args.data_path, args.dataset)
    dataset = np.load(os.path.join(data_path, "X.npy"))

    if args.num_features > 0:
        dataset = dataset[: args.num_features]

    print(f"Dataset shape: {dataset.shape}")

    print("Normalizing dataset...")
    dataset = normalizeDataset(dataset)
    dataset = np.asarray(dataset, dtype=np.float32)

    dimension = dataset.shape[1]

    print("Creating hash function...")
    hasher = BloomHashFunction(
        dimension,
        args.num_hashes,
        args.hash_bits,
        args.threshold,
        args.verbose,
    )

    print("Hashing all vectors...")
    all_hashes = [hasher(dataset[i]) for i in range(dataset.shape[0])]

    print("Computing pooling matrix...")
    pooling = computePools(
        dataset.shape[0],
        args.num_neighbors,
        args.verbose,
    )

    print("Building Orion index...")
    start = time.time()

    index = OrionIndex(
        1 << args.hash_bits,
        args.num_hashes,
        args.threshold,
    )

    index.build(
        all_hashes,
        pooling.items_to_pools,
    )

    elapsed = time.time() - start

    print(f"Index built in {elapsed:.3f} seconds.")
    print("Sleeping for 120 seconds...")
    time.sleep(120)