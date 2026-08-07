import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = "plots"
OUTPUT_DIR = os.path.join("plots", "histograms")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process every CSV in plots/
for csv_file in glob.glob(os.path.join("results", "*.csv")):
    # Read CSV
    df = pd.read_csv(csv_file)

    # Skip files without the required column
    if "KHAN Time" not in df.columns:
        print(f"Skipping {csv_file}: 'KHAN Time' column not found.")
        continue

    # Convert seconds -> milliseconds
    times_ms = df["KHAN Time"] * 1000

    # 10 ms bins
    bins = np.arange(0, times_ms.max() + 10, 0.1)

    plt.figure(figsize=(8, 5))
    plt.hist(times_ms, bins=bins, edgecolor="black")

    plt.xlabel("KHAN Search Time (ms)")
    plt.ylabel("Frequency")

    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    title = f"{base_name}_histogram"
    plt.title(title)

    output_path = os.path.join(OUTPUT_DIR, f"{title}.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved histogram to {output_path}")

print("Done.")