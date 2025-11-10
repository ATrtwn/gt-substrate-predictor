import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

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

def plot_molecular_property_distribution(df_substrate):
    """Plot Pairplot of molecular properties."""
    # --- Compute molecular descriptors ---
    def compute_properties(smiles):
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

def plot_cluster_sizes(cluster_df, cluster_col="cluster_id"):
    """Plot number of sequences per MMseqs2 cluster."""

def plot_split_statistics(splits_dict):
    """Visualize unique enzymes/substrates and label distribution per split."""

def visualize_structure(pdb_file, highlight_residues=None):
    """Open PDB in Py3Dmol / ChimeraX for 3D visualization"""
    pass