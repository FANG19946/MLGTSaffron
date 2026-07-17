#!/usr/bin/env python3
import math, cmath, random, statistics
import os, sys, gc, time
import matplotlib.pyplot as plt
import numpy as np
from numpy import array, ndarray, random as npr, linalg
import csv
from tqdm import tqdm
from typing import List, Tuple, Dict, Set, Literal, Optional, Callable, Iterable, Union, Any
from argparse import ArgumentParser
from data_dir import DATASETS_DIR

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)
DATASETS = ["imagenet", "imdb_wiki", "insta_1m", "mirflickr"]

from mlgt_saffron import  MLGTSaffron, BloomGroupTestingSaffron, MLGTGlobal, OdysseyMLGT

# Global Constant
CURRENT_DATASET = None
ALGO_NAME = None

# Plotting Search Time Bar chart
def plot_saffron_search_times(
    saffron_times: List[float],
    dataset_name: str,
    algo_name: str,
):
    """Save a bar chart of Saffron search times."""

    os.makedirs("plots", exist_ok=True)

    queries = range(1, len(saffron_times) + 1)
    times_ms = [t * 1000 for t in saffron_times]

    plt.figure(figsize=(12, 5))
    plt.bar(queries, times_ms)

    plt.xlabel("Query Number")
    plt.ylabel("Search Time (ms)")
    plt.title(f"Saffron Search Times - {dataset_name}- {algo_name}")
    plt.grid(axis="y", alpha=0.3)

    filename = os.path.join(
        "plots",
        f"saffron_search_times_{dataset_name.lower()}_{algo_name.lower()}.png"
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved plot to {filename}")

def save_saffron_search_times_csv(
    saffron_times: List[float],
    hash_times: List[float],
    decode_times: List[float],
    dataset_name: str,
    algo_name: str,
):
    """Save Saffron timings to a CSV."""

    os.makedirs("results", exist_ok=True)

    filename = os.path.join(
        "results",
        f"saffron_search_times_{dataset_name.lower()}_{algo_name.lower()}.csv"
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Query", "Saffron Time", "Hash Time", "Decode Time"])

        for i, (s, h, d) in enumerate(zip(saffron_times, hash_times, decode_times), start=1):
            writer.writerow([i, s, h, d])

    print(f"Saved CSV to {filename}")




def test_saffron(
        dataset: ndarray,
        query_set: ndarray,
        num_neighbors: int,
        algo_name: str = "mlgt",
        num_hashes: int = 50,
        hash_bits: int = 20,
        threshold: int = 20,
        verbose: int = 0,
) -> Tuple[float, float, float, float, float, float, float]:
    """
    Test the Saffron implementation on a given dataset and query set.

    Parameters:
    - dataset: ndarray of shape (num_items, num_features)
    - query_set: ndarray of shape (num_queries, num_features)
    - num_neighbors: int, number of nearest neighbors to retrieve
    - algo_name: str, "mlgt" or "bloom"
    - num_hashes: int, number of compound hash functions
    - hash_bits: int, number of bits per compound hash
    - threshold: int, number of matches required

    Returns:
    - Indexing time (for the dataset)
    - Average search time per query
    - Average naive search time per query
    - Average precision per query
    - Average recall per query
    """
    idx_start = time.time()
    if algo_name == "mlgt":
        saffron_index = MLGTSaffron(
            dataset, num_neighbors, 
            num_hashes=num_hashes, hash_bits=hash_bits, threshold=threshold,
            debug=verbose
        ) # type: ignore
    elif algo_name == "global":
        saffron_index = MLGTGlobal(
            dataset, num_neighbors, 
            num_hashes=num_hashes, hash_bits=hash_bits, threshold=threshold,
            debug=verbose
        ) # type: ignore
    elif algo_name == "bloom":
        saffron_index = BloomGroupTestingSaffron(
            dataset, num_neighbors,
            num_hashes=num_hashes, hash_bits=hash_bits, threshold=threshold,
            debug=verbose
        ) # type: ignore
    elif algo_name == "odyssey":
        saffron_index = OdysseyMLGT(
        dataset, num_neighbors, 
        num_hashes=num_hashes, hash_bits=hash_bits, threshold=threshold,
        debug=verbose
    ) # type: ignore
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    idx_time: float = time.time() - idx_start
    total_saffron_time: float = 0.0
    total_naive_time: float = 0.0
    total_precision: float = 0.0
    total_recall: float = 0.0
    num_queries: int = query_set.shape[0]
    # List of Saffron and Search times
    saffron_times: List[float] = []
    hash_times: List[float] = []
    decode_times: List[float] = []
    

    # Add Hash and Decode Times
    total_hash_time: float = 0.0
    total_decode_time: float = 0.0
    
    for qidx in tqdm(range(num_queries), desc="Testing queries"):
        query: ndarray = query_set[qidx]
        
        # Saffron search
        start_time: float = time.time()
        # retrieved_indices: List[int] = saffron_index.search(query) # type: ignore
        # Updating saffron search with stats
        result = saffron_index.search(query)

        # Had to Add this to recieve hash and decode time of MLGTSaffron because its not supported for other methods
        if isinstance(result, tuple):
            retrieved_indices, hashing_time, decoding_time = result
        else:
            retrieved_indices = result
            hashing_time = 0.0
            decoding_time = 0.0



        saffron_time: float = time.time() - start_time
        # Appending to List of Times
        saffron_times.append(saffron_time)
        hash_times.append(hashing_time)
        decode_times.append(decoding_time)

        
        # Naive search
        start_time = time.time()
        distances: ndarray = dataset @ query
        true_indices: ndarray = np.argsort(distances)[-num_neighbors:]
        naive_time: float = time.time() - start_time
        
        # Compute precision and recall
        retrieved_set: Set[int] = set(retrieved_indices)
        true_set: Set[int] = set(true_indices)
        
        true_positives: int = len(retrieved_set.intersection(true_set))
        precision: float = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall: float = true_positives / len(true_set) if true_set else 0.0

        if (verbose > 0):
            print(f"Query {qidx + 1}/{num_queries}:")
            print(f"  Saffron retrieved indices: {retrieved_indices}")
            print(f"    Dot products: {distances[retrieved_indices].tolist()}")
            print(f"  True nearest indices: {true_indices.tolist()}")
            print(f"    Dot products: {distances[true_indices].tolist()}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Saffron time: {saffron_time:.6f} s, Naive time: {naive_time:.6f} s")
        
        # Aggregate results
        total_saffron_time += saffron_time
        total_naive_time += naive_time
        total_precision += precision
        total_recall += recall
        
        # Total Hash and Decode Times
        total_hash_time += hashing_time
        total_decode_time += decoding_time
    
    avg_saffron_time: float = total_saffron_time / num_queries
    avg_naive_time: float = total_naive_time / num_queries
    avg_precision: float = total_precision / num_queries
    avg_recall: float = total_recall / num_queries

    # Total Hash and Decode Times
    avg_hash_time: float = total_hash_time / num_queries
    avg_decode_time: float = total_decode_time / num_queries
    # Plot Histogram 
    # plot_saffron_search_times(saffron_times, CURRENT_DATASET, ALGO_NAME)
    # Save CSV
    save_saffron_search_times_csv( saffron_times, hash_times, decode_times, CURRENT_DATASET, ALGO_NAME)
    return idx_time, avg_saffron_time, avg_naive_time, avg_precision, avg_recall, avg_hash_time, avg_decode_time



if __name__ == "__main__":
    parser = ArgumentParser("Test Saffron implementation")
    parser.add_argument(
        "--data-path",
        "-p",
        type=str, 
        default=DATASETS_DIR,
        help="Path to the datasets' directory [default: DATASETS_DIR]"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        nargs="+",
        default=["all"],
        help="The dataset name(s) [imagenet(default), imdb_wiki, insta_1m, mirflickr, all]"
    )
    parser.add_argument(
        "--num-features",
        "-n",
        type=int,
        default=-1,
        help="Number of dataset features to use (default: all)"
    )
    parser.add_argument(
        "--num-queries",
        "-q",
        type=int, 
        default=1000, 
        help="Number of queries to test (default: all)"
    )
    parser.add_argument(
        "--num-neighbors",
        "--k-val",
        "-k",
        type=int, 
        default=10, 
        help="Number of nearest neighbors to retrieve (default: 10)"
    )
    parser.add_argument(
        "--algo",
        "-a",
        type=str,
        nargs="+",
        choices=["mlgt", "bloom", "global","odyssey"],
        default=["odyssey", "global"],
        help="The algorithm(s) to test [mlgt, bloom, global(default)]"
    )
    parser.add_argument(
        "--num-hashes",
        type=int,
        nargs="+",
        default=[100],
        help="Number of hashes for Bloom Filter (default: 100)"
    )
    parser.add_argument(
        "--hash-bits",
        type=int,
        nargs="+",
        default=[16],
        help="Bits per hash for Bloom Filter (default: 16)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        nargs="+",
        default=[10],
        help="Match threshold for Bloom Filter (default: 10)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=0,
        help="Verbosity level (default: 0)"
    )
    args = parser.parse_args()

    # Determine which datasets to run
    datasets_to_run = args.dataset
    if "all" in datasets_to_run:
        datasets_to_run = DATASETS

    # Nested loops for parameters
    for dname in datasets_to_run:
        CURRENT_DATASET = dname
        # Load data for this dataset
        data_path: str = os.path.join(args.data_path, dname)
        try:
            full_dataset: ndarray = np.load(os.path.join(data_path, "X.npy"))
            full_query_set: ndarray = np.load(os.path.join(data_path, "Q.npy"))
        except FileNotFoundError:
            print(f"\n[Error] Dataset files not found in {data_path}. Skipping.")
            continue

        if args.num_features > 0:
            full_dataset = full_dataset[:args.num_features, :]
        if args.num_queries > 0:
            full_query_set = full_query_set[:args.num_queries, :]

        for algo in args.algo:
            ALGO_NAME = algo
            for nh, hb, th in zip(args.num_hashes, args.hash_bits, args.threshold):
                print(f"\n>>> Running: Dataset={dname}, Algo={algo}, k={args.num_neighbors}, hashes={nh}, bits={hb}, threshold={th}")
                

                idx_time, avg_saffron_time, avg_naive_time, avg_precision, avg_recall, avg_hash_time, avg_decode_time = test_saffron(
                    full_dataset, 
                    full_query_set, 
                    args.num_neighbors,
                    algo,
                    nh,
                    hb,
                    th,
                    args.verbose
                )

                print(f"--- Results for {algo.upper()} on {dname.upper()} ---")
                print(f"Indexing Time: {idx_time:.6f} seconds")
                print(f"Avg Saffron Search: {avg_saffron_time:.6f} seconds")
                print(f"Avg Naive Search:   {avg_naive_time:.6f} seconds")
                print(f"Avg Precision:      {avg_precision:.4f}")
                print(f"Avg Recall:         {avg_recall:.4f}")
                print(f"Avg Hash Time:     {avg_hash_time:.6f} seconds")
                print(f"Avg Decode Time:   {avg_decode_time:.6f} seconds")

