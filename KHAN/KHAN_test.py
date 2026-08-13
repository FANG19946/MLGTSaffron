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

from KHAN import  KHAN, BruteForceHashSearch

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

def save_KHAN_search_times_csv(
    # new logging parameters
    KHAN_times: List[float],
    naive_times: List[float],
    hash_times: List[float],
    probe_times: List[float],
    verification_times: List[float],
    false_candidate_rates: List[float],
    postings_traversed: List[float],
    verified_counts: List[float],
    rejected_counts: List[float],
    dataset_name: str,
    algo_name: str,
):
    """Save KHAN timings to a CSV."""

    os.makedirs("results", exist_ok=True)

    filename = os.path.join(
        "results",
        f"KHAN_search_times_{dataset_name.lower()}_{algo_name.lower()}.csv"
    )

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        # new logging parameters
        writer.writerow(["Query", "KHAN Time", "Naive Times", "Hash Time", "Probe Time", "Verification Time", "Candidate Rejection Rate", "Postings Traversed", "Verified Counts", "Rejected Counts"])

        for i, (k, n, h, p, v, fp, pt, vc, rc) in enumerate(zip(KHAN_times, naive_times, hash_times, probe_times, verification_times, false_candidate_rates, postings_traversed, verified_counts, rejected_counts), start=1):
            writer.writerow([i, k, n, h, p, v, fp, pt, vc, rc])

    print(f"Saved CSV to {filename}")




def test_KHAN(
        dataset: ndarray,
        query_set: ndarray,
        num_neighbors: int,
        algo_name: str = "mlgt",
        num_hashes: int = 50,
        hash_bits: int = 20,
        threshold: int = 20,
        verbose: int = 0,
        # new logging parameters
) -> Tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    #region description
    """
    Test the KHAN implementation on a given dataset and query set.

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
    - Average search time per query using KHAN
    - Average naive search time per query
    - Average precision per query
    - Average recall per query
    - Average Hashing times per query
    - Average Probing times per query
    - Average Verification times per query
    - Average False Candidate Rate per query
    - Average Postings Traversed per query
    - Average Verified Counts per query
    - Average Rejected Counts per query
    """
    # endregion
    
    idx_start = time.time()
    
    if algo_name == "KHAN":
        KHAN_index = KHAN(
            dataset,
            num_neighbors,
            dataset.shape[0],
            num_hashes=num_hashes,
            hash_bits=hash_bits,
            debug=verbose,
        ) # type: ignore
        brute_index = BruteForceHashSearch(
            KHAN_index.threshold_,
            dataset.shape[0],
            dataset.shape[1],
            num_hashes,
            hash_bits, 
            verbose,
            KHAN_index.all_hashes_,
            KHAN_index.packed_hashes_
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    # new logging parameters
    # region timer initializations
    idx_time: float = time.time() - idx_start
    total_KHAN_time: float = 0.0
    total_false_candidate_rate = 0.0
    total_naive_time: float = 0.0
    total_precision: float = 0.0
    total_recall: float = 0.0
    total_probe_time: float = 0.0
    total_verification_time: float = 0.0
    total_postings_traversed: float = 0.0
    total_verified_counts: float = 0.0
    total_rejected_counts: float = 0.0
    
    num_queries: int = query_set.shape[0]
    # List of KHAN and Search times
    # new logging parameters
    KHAN_times: List[float] = []
    hash_times: List[float] = []
    probe_times: List[float] = []
    verification_times: List[float] = []
    naive_times: List[float] = []
    false_candidate_rates: List[float] = []
    postings_traversed_list: List[float] = []
    verified_counts_list: List[float] = []
    rejected_counts_list: List[float] = []  


    # endregion
    

    # Add Hash Times
    total_hash_time: float = 0.0
    
    
    for qidx in tqdm(range(num_queries), desc="Testing queries"):
        query: ndarray = query_set[qidx]
        
        # KHAN search
        start_time: float = time.time()
        # retrieved_indices: List[int] = KHAN_index.search(query) 
        result = KHAN_index.search(query)
        


        
        if isinstance(result, tuple):
            retrieved_indices, hashing_time, probing_time, verification_time, false_candidate_rate, postings_traversed, verified_count, rejected_count = result
        else:
            retrieved_indices = result
            hashing_time = 0.0
            probing_time = 0.0
            verification_time = 0.0
            false_candidate_rate = 0.0
            postings_traversed = 0.0
            verified_count = 0.0
            rejected_count = 0.0


        KHAN_time: float = time.time() - start_time
        # Appending to List of Times
        # new logging parameters
        KHAN_times.append(KHAN_time)
        hash_times.append(hashing_time)
        probe_times.append(probing_time)
        verification_times.append(verification_time)
        false_candidate_rates.append(false_candidate_rate)
        postings_traversed_list.append(postings_traversed)
        verified_counts_list.append(verified_count)
        rejected_counts_list.append(rejected_count)


        
        # Naive search
        true_indices, naive_time = KHAN_index.bruteSearch(query)
        naive_postings = 0
        naive_times.append(naive_time)
        
        
        # Compute precision and recall
        retrieved_set: Set[int] = set(retrieved_indices)
        true_set: Set[int] = set(true_indices)
        
        true_positives: int = len(retrieved_set.intersection(true_set))
        precision: float = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall: float = true_positives / len(true_set) if true_set else 0.0
        if not retrieved_set and not true_set:
            precision = 1.0
            recall = 1.0

        false_positives = sorted(retrieved_set - true_set)
        false_negatives = sorted(true_set - retrieved_set)

        if (verbose > 0):
            print(f"Query {qidx + 1}/{num_queries}:")
            print(f"  False positives ({len(false_positives)}): {false_positives}")
            print(f"  False negatives ({len(false_negatives)}): {false_negatives}")
            # print(f"  KHAN retrieved indices: {retrieved_indices}")
            # print(f"    Dot products: {distances[retrieved_indices].tolist()}")
            # print(f"  True nearest indices: {list(true_indices)}")
            # print(f"    Dot products: {distances[true_indices].tolist()}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            # print(f"  KHAN time: {KHAN_time:.6f} s, Naive time: {naive_time:.6f} s")
        
        # Aggregate results
        total_KHAN_time += KHAN_time
        total_naive_time += naive_time
        total_precision += precision
        total_recall += recall
        
        # Total Hash and Decode Times
        total_hash_time += hashing_time
        total_probe_time += probing_time
        total_verification_time += verification_time
        total_false_candidate_rate += false_candidate_rate
        total_postings_traversed += postings_traversed
        total_verified_counts += verified_count
        total_rejected_counts += rejected_count

    avg_KHAN_time: float = total_KHAN_time / num_queries
    avg_naive_time: float = total_naive_time / num_queries
    avg_precision: float = total_precision / num_queries
    avg_recall: float = total_recall / num_queries

    # Total Hash and Decode Times
    avg_hash_time: float = total_hash_time / num_queries
    avg_probe_time: float = total_probe_time / num_queries
    avg_verification_time: float = total_verification_time / num_queries
    avg_false_candidate_rate: float = total_false_candidate_rate / num_queries
    avg_postings_traversed: float = total_postings_traversed / num_queries
    avg_verified_counts: float = total_verified_counts / num_queries
    avg_rejected_counts: float = total_rejected_counts / num_queries

    # Plot Histogram 
    # plot_saffron_search_times(saffron_times, CURRENT_DATASET, ALGO_NAME)
    # Save CSV
    save_KHAN_search_times_csv( KHAN_times, naive_times, hash_times, probe_times, verification_times, false_candidate_rates, postings_traversed_list, verified_counts_list, rejected_counts_list, CURRENT_DATASET, ALGO_NAME)
    return (
    idx_time,
    avg_KHAN_time,
    avg_naive_time,
    avg_precision,
    avg_recall,
    avg_hash_time,
    avg_probe_time,
    avg_verification_time,
    avg_false_candidate_rate,
    avg_postings_traversed,
    avg_verified_counts,
    avg_rejected_counts
    )



if __name__ == "__main__":
    parser = ArgumentParser("Test KHAN implementation")
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
        choices=["KHAN"],
        default=["KHAN"],
        help="The algorithm(s) to test [mlgt, bloom, global, odyssey(default)]"
    )
    parser.add_argument(
        "--num-hashes",
        type=int,
        nargs="+",
        default=[512],
        help="Number of hashes for Bloom Filter (default: 100)"
    )
    parser.add_argument(
        "--hash-bits",
        type=int,
        nargs="+",
        default=[1],
        help="Bits per hash for Bloom Filter (default: 16)"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        nargs="+",
        default=[20],
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
                

                idx_time, avg_KHAN_time, avg_naive_time, avg_precision, avg_recall, avg_hash_time, avg_probe_time, avg_verification_time, avg_false_candidate_rate, avg_postings_traversed, avg_verified_counts, avg_rejected_counts = test_KHAN(
                    full_dataset, 
                    full_query_set, 
                    args.num_neighbors,
                    algo,
                    nh,
                    hb,
                    th,
                    args.verbose
                )

                # print(f"--- Results for {algo.upper()} on {dname.upper()} ---")
                # print(f"Indexing Time: {idx_time:.6f} seconds")
                # print(f"Avg KHAN Search: {avg_KHAN_time:.6f} seconds")
                # print(f"Avg Naive Search:   {avg_naive_time:.6f} seconds")
                # print(f"Avg Precision:      {avg_precision:.4f}")
                # print(f"Avg Recall:         {avg_recall:.4f}")
                # print(f"Avg Hash Time:     {avg_hash_time:.6f} seconds")
                # print(f"Avg Decode Time:   {avg_decode_time:.6f} seconds")
                # print(f"Avg Test Evaluation Time:   {avg_test_evaluation_time:.6f} seconds")

                with open("KHAN_results.txt", "a") as f:

                    def log(msg):
                        print(msg)          # terminal
                        print(msg, file=f)  # file

                    log(f"--- Results for {algo.upper()} on {dname.upper()} ---")
                    log(f"Indexing Time: {idx_time:.6f} seconds")
                    log(f"Avg KHAN Search: {avg_KHAN_time:.6f} seconds")
                    log(f"Avg Naive Search:   {avg_naive_time:.6f} seconds")
                    log(f"Avg Precision:      {avg_precision:.4f}")
                    log(f"Avg Recall:         {avg_recall:.4f}")
                    log(f"Avg Hash Time:      {avg_hash_time:.6f} seconds")
                    log(f"Avg Probe Time:    {avg_probe_time:.6f} seconds")
                    log(f"Avg Verification Time: {avg_verification_time:.6f} seconds")
                    log(f"Avg False Candidate Rate: {avg_false_candidate_rate:.6f} ")
                    log("")


