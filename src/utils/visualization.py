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

def plot_split_statistics(splits_dict):
    """Visualize unique enzymes/substrates and label distribution per split."""

def visualize_structure(pdb_file, highlight_residues=None):
    """Open PDB in Py3Dmol / ChimeraX for 3D visualization"""
    pass