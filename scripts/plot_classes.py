
import os
import pandas as pd
from pathlib import Path
import sys
# Ensure project root is in sys.path for src imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.visualization import (
    plot_class_balance,
    plot_sequence_length_distribution,
    plot_molecular_property_distribution
)
from src.data.preprocessing import binarize_activity


# data directory
data_dir = Path(__file__).parent.parent / "data"
FULL_DATASET_FILE = os.path.join(data_dir, "full_dataset.csv")
UGT_FILE = os.path.join(data_dir, "UGT.csv")
SUBSTRATE_FILE = os.path.join(data_dir, "Substrate.csv")

# directory for plots
OUTPUT_DIR = "../reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Load CSVs
    df_ugt = pd.read_csv(UGT_FILE)
    df_substrate = pd.read_csv(SUBSTRATE_FILE)
    df_full = pd.read_csv(FULL_DATASET_FILE)

    # Merge full dataset with gt/substrate info if needed
    if 'prot_seq' not in df_full.columns:
        df_merged = df_full.merge(df_ugt, left_on="UGT_trivial_name", right_on="UGT_trivial_name", how="left")
    else:
        df_merged = df_full
    if 'smiles' not in df_merged.columns:
        df_merged = df_merged.merge(df_substrate, left_on="substrate", right_on="substrate", how="left")

    # Active vs Inactive (is_active)
    plot_class_balance(df_merged, label_col="is_active")
    # gt lengths
    plot_sequence_length_distribution(df_ugt, seq_col="prot_seq")
    # Molecular property distributions (use full dataset and correct SMILES column)
    plot_molecular_property_distribution(df_merged, smiles_col="SMILES_isomeric_1")
    print("Visualizations completed. Check out reports folder.")


if __name__ == "__main__":
    main()