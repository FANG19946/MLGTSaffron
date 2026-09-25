import time
import numpy as np
import flinng
import argparse
import os
import sys
from tqdm import tqdm

CUR_DIR: str = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

DATASETS = [ "imagenet", "imdb_wiki", "insta_1m", "mirflickr", "imagenet_penultimate",]

CURRENT_DATASET = None
ALGO_NAME = "FLINNG"

DATASET_DIR = "/mnt/Drive1/Datasets_NNSearch"


parser = argparse.ArgumentParser()

parser.add_argument("-n", "--num-items", type=int, default=None)
parser.add_argument("-q", "--num-queries", type=int, default=None)
parser.add_argument("-k", "--top-k", type=int, default=10000)
parser.add_argument("--degrees", type=float, default=10.0)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    nargs="+",
    default=["imagenet"],
    help="The dataset name(s) [imagenet(default), imdb_wiki, insta_1m, mirflickr, imagenet_penultimate, all]"
)

args = parser.parse_args()

if "all" in args.dataset:
    datasets = DATASETS
else:
    datasets = args.dataset

for dataset_name in datasets:
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

# =========================
# Dataset
# =========================
for dataset_name in datasets:
    CURRENT_DATASET = dataset_name

    X_PATH = os.path.join(DATASET_DIR, CURRENT_DATASET, "X.npy")
    Q_PATH = os.path.join(DATASET_DIR, CURRENT_DATASET, "Q.npy")

  


    # =========================
    # Experiment parameters
    # =========================

    ANGLE_DEGREES = args.degrees
    TOP_K = args.top_k


    NUM_ROWS = 3
    CELLS_PER_ROW = 10000
    HASHES_PER_TABLE = 16
    NUM_HASH_TABLES = 20

    # =========================
    # Load data
    # =========================

    dataset = np.load(X_PATH).astype(np.float32)
    queries = np.load(Q_PATH).astype(np.float32)

    if args.num_items is not None:
        dataset = dataset[:args.num_items]

    if args.num_queries is not None:
        queries = queries[:args.num_queries]
    print(f"\n")
    # print(f"Dataset: {dataset.shape}")
    # print(f"Queries: {queries.shape}")

    # Normalize for cosine similarity
    dataset_norm = dataset / np.linalg.norm(dataset, axis=1, keepdims=True)
    queries_norm = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    # =========================
    # Build FLINNG index
    # =========================

    start = time.perf_counter()

    index = flinng.dense_32_bit(
        num_rows=NUM_ROWS,
        cells_per_row=CELLS_PER_ROW,
        data_dimension=dataset.shape[1],
        num_hash_tables=NUM_HASH_TABLES,
        hashes_per_table=HASHES_PER_TABLE,
    )

    index.add_points(dataset)
    index.prepare_for_queries()

    index_time = time.perf_counter() - start

    # print(f"Index time: {index_time:.4f} s")

    # =========================
    # FLINNG query
    # =========================

    start = time.perf_counter()

    results = index.query(queries, TOP_K)

    flinng_time = time.perf_counter() - start
    avg_flinng_time = flinng_time / len(queries)

    # print(f"Average FLINNG query time: {avg_flinng_time:.6f} s/query")


    # Verify the 10K candidates against the 10-degree threshold
    start = time.perf_counter()

    cosine_threshold = np.cos(np.deg2rad(ANGLE_DEGREES))

    verified_results = []

    for i, query in enumerate(queries_norm):
        retrieved = np.asarray(results[i])

        # Compute exact cosine similarity only for the 10K retrieved candidates
        similarities = dataset_norm[retrieved] @ query

        # Keep only candidates actually within 10 degrees
        verified = retrieved[similarities >= cosine_threshold]

        verified_results.append(verified)

    verification_time = time.perf_counter() - start
    avg_verification_time = verification_time / len(queries)

    # print(f"Average verification time: {avg_verification_time:.6f} s/query")

    total_query_time = flinng_time + verification_time
    avg_query_time = total_query_time / len(queries)

    # print(f"Average query time: {avg_query_time:.6f} s/query")

    # =========================
    # Evaluate FLINNG results
    # =========================

    cosine_threshold = np.cos(np.deg2rad(ANGLE_DEGREES))

    total_candidates = 0
    total_verified = 0
    total_true = 0
    total_correct = 0

    total_precision = 0.0
    total_recall = 0.0

    # Exact ground truth + evaluation
    evaluation_start = time.perf_counter()

    for i, query in tqdm(enumerate(queries_norm), total=len(queries_norm), desc="Evaluating"):

        # True 10-degree neighbors
        similarities = dataset_norm @ query
        true_neighbors = np.where(similarities >= cosine_threshold)[0]
        true_set = set(true_neighbors)

        # FLINNG candidates
        retrieved = np.asarray(results[i])

        # Keep only candidates actually inside the 10-degree cap
        retrieved_similarities = similarities[retrieved]
        retrieved_neighbors = retrieved[
            retrieved_similarities >= cosine_threshold
        ]

        retrieved_set = set(retrieved_neighbors)

        true_positives = len(retrieved_set.intersection(true_set))

        precision = (
            true_positives / len(retrieved_set)
            if retrieved_set
            else 1.0
        )

        recall = (
            true_positives / len(true_set)
            if true_set
            else 1.0
        )

        if not retrieved_set and not true_set:
            precision = 1.0
            recall = 1.0

        total_true += len(true_set)
        total_candidates += len(retrieved)
        total_verified += len(retrieved_neighbors)
        total_correct += true_positives

        total_precision += precision
        total_recall += recall

        # if (i + 1) % 100 == 0:
        #     print(f"Evaluated {i + 1}/{len(queries)} queries")

    evaluation_time = time.perf_counter() - evaluation_start

    # =========================
    # Precision / Recall
    # =========================

    precision = total_precision / len(queries)
    recall = total_recall / len(queries)

    # print()
    # print(f"10-degree precision: {precision:.6f}")
    # print(f"10-degree recall:    {recall:.6f}")
    # print(f"Average query time:  {avg_query_time:.6f} s/query")
    # print(f"Average true neighbors/query: {total_true / len(queries):.2f}")
    # print(f"Average candidates retrieved/query: {total_candidates / len(queries):.2f}")
    # print(f"Average verified neighbors/query: {total_verified / len(queries):.2f}")
    # print(f"Evaluation time: {evaluation_time:.4f} s \n\n")


    print()
    # print(f"{CURRENT_DATASET.upper()}")
    print(f"--- Results for {ALGO_NAME} on {CURRENT_DATASET.upper()} ---\n")

    # print(f"--- Results for {ALGO_NAME.upper()} ---")
    print(f"Indexing Time: {index_time:.6f} seconds")
    print(f"Number of Items in Dataset: {len(dataset)}")
    print(f"Number of Queries Tested: {len(queries)}")
    print(f"Number of Degrees: {ANGLE_DEGREES:.1f}")

    print(f"Number of Intersections: {NUM_ROWS}")
    print(f"Number of Tests per Repetition: {CELLS_PER_ROW}")
    # print(f"Num Hashes: {HASHES_PER_TABLE}")
    print(f"Num Bits: {HASHES_PER_TABLE}")
    print(f"Number of Hash Tables: {NUM_HASH_TABLES}")
    print(f"Top-K Candidates Retrieved: {TOP_K}")

    print(f"Avg FLINNG Query Time: {avg_query_time:.6f} seconds")
    print(f"Avg FLINNG Candidate Generation Time: {avg_flinng_time:.6f} seconds")
    print(f"Avg Verification Time: {avg_verification_time:.6f} seconds")

    print(f"Avg Precision: {precision:.4f}")
    print(f"Avg Recall: {recall:.4f}")

    print(f"Avg True Neighbors: {total_true / len(queries):.2f}")
    print(f"Avg Candidates Retrieved: {total_candidates / len(queries):.2f}")
    print(f"Avg Verified Neighbors: {total_verified / len(queries):.2f}\n\n\n")

    # =========================
    # Save results
    # =========================

    # with open("FLINNG_results.txt", "a") as f:
    #     f.write("\n")
    #     f.write("=" * 60 + "\n")
    #     f.write(f"Dataset: {X_PATH}\n")
    #     f.write(f"Dataset shape: {dataset.shape}\n")
    #     f.write(f"Queries shape: {queries.shape}\n")
    #     f.write(f"Angle cap: {ANGLE_DEGREES} degrees\n")
    #     f.write(f"TOP_K: {TOP_K}\n")
    #     f.write(f"NUM_ROWS: {NUM_ROWS}\n")
    #     f.write(f"CELLS_PER_ROW: {CELLS_PER_ROW}\n")
    #     f.write(f"HASHES_PER_TABLE: {HASHES_PER_TABLE}\n")
    #     f.write(f"NUM_HASH_TABLES: {NUM_HASH_TABLES}\n")
    #     f.write(f"Index time: {index_time:.4f} s\n")
    #     f.write(f"Average query time: {avg_query_time:.6f} s/query\n")
    #     f.write(f"Precision: {precision:.6f}\n")
    #     f.write(f"Recall: {recall:.6f}\n")
    #     f.write(
    #         f"Average true neighbors/query: "
    #         f"{total_true / len(queries):.2f}\n"
    #     )
    #     f.write(
    #         f"Average retrieved neighbors/query: "
    #         f"{total_retrieved / len(queries):.2f}\n"
    #     )

    with open("FLINNG_results.txt", "a") as f:
        f.write("\n")
        f.write("=" * 60 + "\n\n\n")
        f.write(f"--- Results for {ALGO_NAME} on {CURRENT_DATASET.upper()} ---\n")
        f.write(f"Indexing Time: {index_time:.4f} seconds\n")

        f.write(f"Number of Items in Dataset: {len(dataset)}\n")
        f.write(f"Number of Queries Tested: {len(queries)}\n")
        f.write(f"Number of Degrees: {ANGLE_DEGREES}\n")

        f.write(f"Number of Intersections: {NUM_ROWS}\n")
        f.write(f"Number of Tests per Repetition: {CELLS_PER_ROW}\n")
        # f.write(f"Num Hashes: {HASHES_PER_TABLE}\n")
        f.write(f"Num Bits: {HASHES_PER_TABLE}\n")
        f.write(f"Number of Hash Tables: {NUM_HASH_TABLES}\n")
        f.write(f"Top-K Candidates Retrieved: {TOP_K}\n")

        f.write(f"Avg FLINNG Query Time: {avg_query_time:.6f} seconds\n")
        f.write(f"Avg FLINNG Candidate Generation Time: {avg_flinng_time:.6f} seconds\n")
        f.write(f"Avg Verification Time: {avg_verification_time:.6f} seconds\n")

        f.write(f"Avg Precision: {precision:.4f}\n")
        f.write(f"Avg Recall: {recall:.6f}\n")

        f.write(f"Avg True Neighbors: {total_true / len(queries):.2f}\n")
        f.write(f"Avg Candidates Retrieved: {total_candidates / len(queries):.2f}\n")
        f.write(f"Avg Verified Neighbors: {total_verified / len(queries):.2f}\n\n\n")