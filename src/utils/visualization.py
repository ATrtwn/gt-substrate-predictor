import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from upsetplot import UpSet, from_indicators

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

def plot_class_balance(df, label_col="activity", label_rot=True):
    """Plot the fractions of active vs. inactive samples."""
    counts = df[label_col].value_counts(normalize=True)
    plt.figure()
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette="colorblind", legend=False)
    plt.title("Class Balance")
    plt.xlabel(label_col)
    if label_rot:
        plt.xticks(rotation=30, ha='right')
    else:
        plt.xticks(ha='right')
    plt.ylabel("Fraction of Samples")
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, f"class_balance_{label_col}.png")
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

def plot_molecular_property_distribution(df_substrate):
    """Plot Pairplot of molecular properties."""
    # --- Compute molecular descriptors ---
    def compute_properties(smiles):
        # Disable RDKit warnings
        RDLogger.DisableLog('rdApp.warning')
        if type(smiles) != str:
            return None
        if smiles is None or smiles.strip() == "":
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        props = {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "NumHDonors": Descriptors.NumHDonors(mol),
            "NumHAcceptors": Descriptors.NumHAcceptors(mol),
            "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        }
        return props
    props_df = df_substrate["ConnectivitySMILES"].apply(compute_properties).dropna().apply(pd.Series)
    g = sns.pairplot(props_df, vars=["MolWt", "LogP", "TPSA","NumHDonors","NumHAcceptors", "NumRotatableBonds"], diag_kind="kde")
    plt.suptitle("Molecular Property Distributions by Binding Activity")
    g.figure.suptitle("Molecular Property Distributions by Binding Activity", y=1.02)
    g.figure.tight_layout(pad=1.5)  # Adjust spacing
    output_path = os.path.join(FIGURES_DIR, "molecular_property_distributions.png")
    plt.savefig(output_path)
    plt.close()

def plot_split_graph(G, train_edges, val_edges, C1_edges, C2_edges, C3_edges, seen_nodes):
    import networkx as nx
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))

    # Node colors
    node_colors = []
    for n, data in G.nodes(data=True):
        is_seen = n in seen_nodes
        is_protein = data.get("bipartite") == "protein"

        if is_protein and is_seen:
            node_colors.append("#4DA6FF") # green
        elif is_protein and not is_seen:
            node_colors.append("#003F7F") # dark green
        elif not is_protein and is_seen:
            node_colors.append("#FF9999") # red
        else:
            node_colors.append("#7F0000") # dark red

    # Edge colors by split
    edge_color_map = []
    for u, v, data in G.edges(data=True):
        e = (u, v, data["label"], data["force_train"])
        if e in train_edges:
            edge_color_map.append("#8ad1e5") # green
        elif e in val_edges:
            edge_color_map.append("#669cac") # darker green
        elif e in C1_edges:
            edge_color_map.append("#ad2d7c") # red
        elif e in C2_edges:
            edge_color_map.append("#85005a") # darker red
        elif e in C3_edges:
            edge_color_map.append("#5f003a") # even darker red
        else:
            edge_color_map.append("#dcdcdc") # grey

    # Bipartite layout
    proteins = [n for n, d in G.nodes(data=True) if d["bipartite"] == "protein"]
    substrates = [n for n in G.nodes() if n not in proteins]
    pos = {**nx.bipartite_layout(G, proteins)}

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        edge_color=edge_color_map,
        with_labels=False,
        alpha=0.8,
        node_size=50,
    )

    plt.title("Graph-based Dataset Split")
    output_path = os.path.join(FIGURES_DIR, "data_split_graph.png")
    plt.savefig(output_path)
    plt.close()

def plot_graph_connectivity(g):
    degrees = {node: g.degree(node) for node in g.nodes()}
    plt.hist(list(degrees.values()), bins=20)
    plt.title("Node degree distribution (number of partners per protein/substrate)")
    plt.xlabel("Degree")
    plt.ylabel("Number of nodes")
    output_path = os.path.join(FIGURES_DIR, "graph_connectivity.png")
    plt.savefig(output_path)
    plt.close()

def plot_connectivity_separate(G):
    # Identify node types from bipartite structure:
    proteins = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "protein"]
    substrates = [n for n, d in G.nodes(data=True) if d.get("bipartite") == "substrate"]

    # Compute degrees
    protein_deg = [G.degree(p) for p in proteins]
    substrate_deg = [G.degree(s) for s in substrates]

    # Plot
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.hist(protein_deg, bins=20)
    plt.title("Protein Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("# Proteins")

    plt.subplot(1,2,2)
    plt.hist(substrate_deg, bins=20)
    plt.title("Substrate Degree Distribution")
    plt.xlabel("Degree")
    plt.ylabel("# Substrates")

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "graph_connectivity_separate.png")
    plt.savefig(output_path)
    plt.close()

def plot_split_statistics(df_split, protein_col, substrate_col, label_col, split_col="split"):
    """Visualize unique enzymes/substrates and label distribution per split."""

    count_stats = []
    label_stats = []
    for split in df_split[split_col].unique():
        subset = df_split[df_split[split_col] == split]
        proteins = set(subset[protein_col].unique())
        substrates = set(subset[substrate_col].unique())
        count_stats.append({
            "split": split,
            "unique_proteins": len(proteins),
            "unique_substrates": len(substrates),
        })
    for split in df_split[split_col].unique():
        subset = df_split[df_split[split_col] == split]
        counts = subset[label_col].value_counts()
        fractions = counts / counts.sum()
        label_stats.append({
            "split": split,
            "active": fractions.get(1, 0.0),
            "inactive": fractions.get(0, 0.0),
        })

    stats_df = pd.DataFrame(count_stats).set_index("split")
    label_df = pd.DataFrame(label_stats).set_index("split")

    # Unique proteins / unique substrates
    stats_df.plot(kind="bar", figsize=(10, 6))
    plt.title("Proteins/Substrates Statistics per Split")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "data_split_stats_component_distr.png")
    plt.savefig(output_path)
    plt.close()

    # Label balance
    label_df.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.ylabel("Fraction of samples")
    plt.title("Label Distribution per Split")
    plt.xticks(rotation=0)
    plt.legend(title=label_col)
    output_path = os.path.join(FIGURES_DIR, "data_split_stats_label_distr.png")
    plt.savefig(output_path)
    plt.close()

def plot_upset_sets(train_df, val_df, c1_df, c2_df, c3_df):
    """
    Creates UpSet plots showing which component nodes appear in:
        - Seen (train + val)
        - C1 (both nodes seen)
        - C2 (one unseen node)
        - C3 (both unseen)
    """

    # Enzymes
    seen = set(train_df["UGT_ID"]) | set(val_df["UGT_ID"])
    c1 = set(c1_df["UGT_ID"])
    c2 = set(c2_df["UGT_ID"])
    c3 = set(c3_df["UGT_ID"])
    all_enzymes = seen | c1 | c2 | c3
    df = pd.DataFrame({
        "UGT_ID": list(all_enzymes),
        "Seen": [e in seen for e in all_enzymes],
        "C1":   [e in c1 for e in all_enzymes],
        "C2":   [e in c2 for e in all_enzymes],
        "C3":   [e in c3 for e in all_enzymes],
    })
    upset_data = from_indicators(
        ["Seen", "C1", "C2", "C3"],
        df[["Seen", "C1", "C2", "C3"]]
    )
    plt.figure(figsize=(12, 6))
    UpSet(upset_data, subset_size="count", show_counts=True).plot()
    plt.title("Enzymes per data split")
    out_path = os.path.join(FIGURES_DIR, "upset_enzymes.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    # substrates
    seen = set(train_df["substrate"]) | set(val_df["substrate"])
    c1 = set(c1_df["substrate"])
    c2 = set(c2_df["substrate"])
    c3 = set(c3_df["substrate"])
    all_enzymes = seen | c1 | c2 | c3
    df = pd.DataFrame({
        "substrate": list(all_enzymes),
        "Seen": [e in seen for e in all_enzymes],
        "C1": [e in c1 for e in all_enzymes],
        "C2": [e in c2 for e in all_enzymes],
        "C3": [e in c3 for e in all_enzymes],
    })
    upset_data = from_indicators(
        ["Seen", "C1", "C2", "C3"],
        df[["Seen", "C1", "C2", "C3"]]
    )
    plt.figure(figsize=(12, 6))
    UpSet(upset_data, subset_size="count", show_counts=True).plot()
    plt.title("Substrates per data split")
    out_path = os.path.join(FIGURES_DIR,"upset_substrate.png"
    )
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_degree_distributions(G):
    degrees_protein = [deg for n, deg in G.degree()
                       if G.nodes[n].get('bipartite') == "protein"]
    degrees_sub = [deg for n, deg in G.degree()
                   if G.nodes[n].get('bipartite') == "substrate"]

    plt.figure(figsize=(8,5))
    sns.histplot(degrees_protein, bins=30, kde=True)
    plt.title("Enzyme Degree Distribution")
    plt.xlabel("Degree")
    output_path = os.path.join(FIGURES_DIR, f"data_enzyme_degree_distr.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    plt.figure(figsize=(8,5))
    sns.histplot(degrees_sub, bins=30, kde=True)
    plt.title("Substrate Degree Distribution")
    plt.xlabel("Degree")
    output_path = os.path.join(FIGURES_DIR, f"data_substrate_degree_distr.png")
    plt.savefig(output_path, dpi=300)
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

def visualize_structure(pdb_file, highlight_residues=None):
    """Open PDB in Py3Dmol / ChimeraX for 3D visualization"""
    pass