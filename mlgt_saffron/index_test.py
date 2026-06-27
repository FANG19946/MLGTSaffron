#!/usr/bin/env python3

import os
import sys
import time
import numpy as np
from argparse import ArgumentParser

from data_dir import DATASETS_DIR

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from mlgt_saffron import MLGTSaffron, BloomGroupTestingSaffron, MLGTGlobal

DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]


if __name__ == "__main__":
    parser = ArgumentParser("Build index only")

    parser.add_argument(
        "--data-path",
        "-p",
        type=str,
        default=DATASETS_DIR,
    )

    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="imagenet",
    )

    parser.add_argument(
        "--num-features",
        "-n",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--algo",
        "-a",
        choices=["mlgt", "global", "bloom"],
        default="mlgt",
    )

    parser.add_argument(
        "--num-neighbors",
        "-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num-hashes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--hash-bits",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    print("Loading dataset...")

    data_path = os.path.join(args.data_path, args.dataset)

    dataset = np.load(os.path.join(data_path, "X.npy"))

    if args.num_features > 0:
        dataset = dataset[:args.num_features]

    print(f"Dataset shape: {dataset.shape}")

    print("Building index...")
    start = time.time()

    if args.algo == "mlgt":
        index = MLGTSaffron(
            dataset,
            args.num_neighbors,
            num_hashes=args.num_hashes,
            hash_bits=args.hash_bits,
            threshold=args.threshold,
            debug=args.verbose,
        )

    elif args.algo == "global":
        index = MLGTGlobal(
            dataset,
            args.num_neighbors,
            num_hashes=args.num_hashes,
            hash_bits=args.hash_bits,
            threshold=args.threshold,
            debug=args.verbose,
        )

    else:
        index = BloomGroupTestingSaffron(
            dataset,
            args.num_neighbors,
            num_hashes=args.num_hashes,
            hash_bits=args.hash_bits,
            threshold=args.threshold,
            debug=args.verbose,
        )

    elapsed = time.time() - start

    print(f"Index built in {elapsed:.3f} seconds.")
    print("Sleeping for 120 seconds...")
    time.sleep(120)