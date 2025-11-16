import os
import matplotlib.pyplot as plt
import seaborn as sns

# set sns style
sns.set_theme(
    context="paper",
    style="whitegrid",
    palette="pastel",
    font="sans-serif",
    font_scale=1.1,
)

# output folder for plots
FIGURES_DIR = "../reports"
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_class_balance(df, label_col="activity"):
    """Plot the counts of active vs. inactive samples."""
    counts = df[label_col].value_counts()
    plt.figure()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="colorblind")
    plt.title("Class Balance")
    plt.xlabel("Activity Class")
    plt.xticks(rotation=30, ha='right')
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "class_balance.png")
    plt.savefig(output_path)
    plt.close()

def plot_sequence_length_distribution(df, seq_col="prot_seq"):
    """Plot histogram of protein sequence lengths."""
    lengths = df[seq_col].dropna().apply(len)
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=30, kde=True, color="skyblue")
    plt.title("Protein Sequence Length Distribution")
    plt.xlabel("Sequence Length (AA)")
    plt.ylabel("Count")
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "sequence_length_distribution.png")
    plt.savefig(output_path)
    plt.close()

def plot_cluster_sizes(cluster_df, cluster_col="cluster_id"):
    """Plot number of sequences per MMseqs2 cluster."""
    # cluster_df expected to have columns [seq_id, rep_id] or [seq_id, cluster_id]
    import pandas as pd
    counts = None
    if cluster_col in cluster_df.columns:
        counts = cluster_df[cluster_col].value_counts()
    else:
        # try common column names
        for c in ("rep_id", "representative", "cluster"):
            if c in cluster_df.columns:
                counts = cluster_df[c].value_counts()
                break
    if counts is None:
        # assume df is mapping seq->rep in two columns
        if cluster_df.shape[1] >= 2:
            counts = cluster_df.iloc[:, 1].value_counts()
        else:
            raise ValueError("cluster_df does not contain cluster information")

    # histogram of cluster sizes
    plt.figure(figsize=(6, 4))
    sns.histplot(counts.values, bins=range(1, max(counts.values) + 2), color="teal")
    plt.yscale('log')
    plt.title("MMseqs2 cluster size distribution")
    plt.xlabel("Cluster size (number of sequences)")
    plt.ylabel("Number of clusters (log scale)")
    plt.tight_layout()
    out1 = os.path.join(FIGURES_DIR, "cluster_size_histogram.png")
    plt.savefig(out1)
    plt.close()

    # cumulative coverage plot
    sorted_counts = counts.sort_values(ascending=False).values
    cumulative = sorted_counts.cumsum() / sorted_counts.sum()
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cumulative) + 1), cumulative, marker='o', markersize=3)
    plt.xscale('log')
    plt.xlabel('Top N clusters (log scale)')
    plt.ylabel('Cumulative fraction of sequences')
    plt.title('Cumulative coverage by top clusters')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    out2 = os.path.join(FIGURES_DIR, "cluster_size_cumulative.png")
    plt.savefig(out2)
    plt.close()

    # Also write top clusters table
    top = counts.sort_values(ascending=False).head(20)
    try:
        top.to_csv(os.path.join(FIGURES_DIR, "top_clusters.tsv"), sep='\t', header=['size'])
    except Exception:
        pass

def plot_split_statistics(splits_dict):
    """Visualize unique enzymes/substrates and label distribution per split."""

def visualize_structure(pdb_file, highlight_residues=None):
    """Open PDB in Py3Dmol / ChimeraX for 3D visualization"""
    pass