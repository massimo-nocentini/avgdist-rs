
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_clustering(df: pl.DataFrame, output_file: str, plot_title: str):
    """
    Plot the distribution of the clustering coefficient.
    """
    #data = df['cc']
    #collected = df.group_by("cc").agg(pl.len().alias("count")).collect()
    collected = df.with_columns(pl.col("cc")).collect()
    data = collected["cc"]
    plt.figure(figsize=(5, 5))
    #plt.hist(data, bins=np.arange(0, 1.1, 0.1), color="blue")
    plt.hist(data, color="blue")
    plt.title(plot_title)
    plt.xlabel("Closeness Centrality")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.grid(linestyle='--', linewidth=0.5)
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    plt.close()
    #print(df.describe())

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <input_file> <output_file> <model_name>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_name = sys.argv[3]
    df = pl.scan_csv(input_file, separator=",", null_values=["nan", "-nan"], schema=dict([("node", pl.Int64),("cc", pl.Float64)]))
    plot_clustering(df, output_file, f"{model_name} Closeness Centrality")
