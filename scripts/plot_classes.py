import os
import pandas as pd
from pathlib import Path

from src.utils.visualization import (
    plot_class_balance,
    plot_sequence_length_distribution
)

# data directory
data_dir = Path(__file__).parent.parent / "data"
ACTIVITY_FILE = os.path.join(data_dir, "Activity.csv")
UGT_FILE = os.path.join(data_dir, "UGT.csv")
SUBSTRATE_FILE = os.path.join(data_dir, "Substrate.csv")

# directory for plots
OUTPUT_DIR = "../reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Load CSVs
    df_ugt = pd.read_csv(UGT_FILE)
    df_substrate = pd.read_csv(SUBSTRATE_FILE)
    df_activity = pd.read_csv(ACTIVITY_FILE)

    # Merge activity with gt/substrate info
    df_merged = df_activity.merge(df_ugt, left_on="UGT_trivial_name", right_on="UGT_trivial_name", how="left")
    df_merged = df_merged.merge(df_substrate, left_on="substrate", right_on="substrate", how="left")

    # Active vs Inactive
    plot_class_balance(df_merged, label_col="activity")
    # gt lengths
    plot_sequence_length_distribution(df_merged, seq_col="prot_seq")

    print("Visualizations completed. Check out reports folder.")


if __name__ == "__main__":
    main()