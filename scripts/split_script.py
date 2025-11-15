import os
import pandas as pd
from pathlib import Path
from src.data.preprocessing import binarize_activity
from src.data.data_split import stratified_split_by_entities, check_split
from src.utils.visualization import plot_split_statistics

# data directory
data_dir = Path(__file__).parent.parent / "data"
ACTIVITY_FILE = os.path.join(data_dir, "Activity.csv")
UGT_FILE = os.path.join(data_dir, "UGT.csv")
SUBSTRATE_FILE = os.path.join(data_dir, "Substrate.csv")

def main():
    # Load CSVs
    df_ugt = pd.read_csv(UGT_FILE)
    df_substrate = pd.read_csv(SUBSTRATE_FILE)
    df_activity = pd.read_csv(ACTIVITY_FILE)

    # Merge activity with gt/substrate info
    df_merged = df_activity.merge(df_ugt, left_on="UGT_trivial_name", right_on="UGT_trivial_name", how="left")
    df_merged = df_merged.merge(df_substrate, left_on="substrate", right_on="substrate", how="left")

    # binarize
    df = binarize_activity(df_merged)

    protein_col = "UGT_trivial_name"
    substrate_col = "substrate"
    label_col = "activity"
    print(f"\nUnique enzymes ({protein_col}): {df[protein_col].nunique()}")
    print(f"Unique substrates ({substrate_col}): {df[substrate_col].nunique()}")
    if label_col in df.columns:
        print(f"\nClass distribution ({label_col}):")
        print(df[label_col].value_counts())

    splits = stratified_split_by_entities(df,
                                          protein_col=protein_col,
                                          substrate_col=substrate_col)

    # check stratification
    c1 = splits['C1']
    c2 = splits['C2']
    c3 = splits['C3']
    train = splits['train']
    val = splits['val']

    dataset_len = len(df[[protein_col, substrate_col]].drop_duplicates())
    print(f"\nOut of {dataset_len} pairs total ")
    print("Class distribution per split:")
    for name, subset in [("Training", train), ("val", val), ("C1", c1), ("C2", c2), ("C3", c3)]:
        counts = subset["is_active"].value_counts(normalize=True).sort_index()
        print(f"{name}: {dict(counts)} (n={len(subset)})")

    print(f"Training: {len(train)} | val: {len(val)} | C1: {len(c1)} | C2: {len(c2)} | C3: {len(c3)}")

    check_split(train, val, c1, c2, c3, protein_col, substrate_col)

    train["split"] = "train"
    val["split"] = "val"
    c1["split"] = "C1"
    c2["split"] = "C2"
    c3["split"] = "C3"
    df_split = pd.concat([train, val, c1, c2, c3], ignore_index=True).reindex(df.index).fillna({"split": "none"})
    plot_split_statistics(df_split, protein_col, substrate_col, label_col="is_active")

if __name__ == "__main__":
    main()