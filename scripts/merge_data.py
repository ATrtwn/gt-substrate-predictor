import os
import pandas as pd
from pathlib import Path
from src.data.preprocessing import binarize_activity
from src.data.data_split import stratified_split_by_entities, check_split
from src.utils.visualization import (
    plot_class_balance,
    plot_sequence_length_distribution,
    plot_molecular_property_distribution,
    plot_split_statistics,
    plot_upset_sets
)

# data directory
data_dir = Path(__file__).parent.parent / "data"

def split_and_analyse_dataset(df_analyse, plots=False):
    print(f"############## Analysis for merged dataset ##############")

    protein_col = "UGT_ID"
    substrate_col = "substrate"
    label_col = "is_active"
    print(f"\nUnique enzymes ({protein_col}): {df_analyse[protein_col].nunique()}")
    print(f"Unique substrates ({substrate_col}): {df_analyse[substrate_col].nunique()}")
    if label_col in df_analyse.columns:
        print(f"\nClass distribution ({label_col}):")
        print(df_analyse[label_col].value_counts())

    df_split = stratified_split_by_entities(df_analyse,
                                          protein_col=protein_col,
                                          substrate_col=substrate_col,
                                          plot=plots)

    # check label distribution
    c1_test = df_split[df_split['split'] == 'C1_test']
    c2_test = df_split[df_split['split'] == 'C2_test']
    c3_test = df_split[df_split['split'] == 'C3_test']
    c1_val = df_split[df_split['split'] == 'C1_val']
    c2_val = df_split[df_split['split'] == 'C2_val']
    c3_val = df_split[df_split['split'] == 'C3_val']
    train = df_split[df_split['split'] == 'train']

    dataset_len = len(df_analyse[[protein_col, substrate_col]].drop_duplicates())
    print(f"\nOut of {dataset_len} distinct pairs ")
    print("Class distribution per split:")
    for name, subset in [("Training", train),
                         ("C1_val", c1_val), ("C2_val", c2_val), ("C3_val", c3_val),
                         ("C1_test", c1_test), ("C2_test", c2_test), ("C3_test", c3_test)]:
        counts = subset["is_active"].value_counts(normalize=True).sort_index()
        print(f"{name}: {dict(counts)} (n={len(subset)})")

    val = pd.concat([c1_val, c2_val, c3_val], axis=0)
    test = pd.concat([c1_test, c2_test, c3_test], axis=0)
    print(f"Training: {len(train)} | val: {len(val)} | test: {len(test)}")

    # check if split is valid
    check_split(train, c1_val, c2_val, c3_val, c1_test, c2_test, c3_test, protein_col, substrate_col)

    # plots
    if plots:
        plot_split_statistics(df_split, protein_col, substrate_col, label_col="is_active", subscript='_merged')
        if len(test) > 1:
            plot_upset_sets(train, val, c1_test, c2_test, c3_test, subscript='_merged')

    # save datasets
    train.to_csv(f"{data_dir}/train.csv", index=False)
    c1_val.to_csv(f"{data_dir}/C1_val.csv", index=False)
    c2_val.to_csv(f"{data_dir}/C2_val.csv", index=False)
    c3_val.to_csv(f"{data_dir}/C3_val.csv", index=False)
    c1_test.to_csv(f"{data_dir}/C1_test.csv", index=False)
    c2_test.to_csv(f"{data_dir}/C2_test.csv", index=False)
    c3_test.to_csv(f"{data_dir}/C3_test.csv", index=False)

def main():

    ### original gt/substrate dataset
    activity = pd.read_csv(os.path.join(data_dir, "Activity.csv"))
    UGT = pd.read_csv(os.path.join(data_dir, "UGT.csv"))
    substrate = pd.read_csv(os.path.join(data_dir, "Substrate.csv"))
    df_original = activity.merge(UGT, left_on="UGT_trivial_name", right_on="UGT_trivial_name", how="left")
    df_original = df_original.merge(substrate, left_on="substrate", right_on="substrate", how="left")
    # binarize
    df_original = binarize_activity(df_original)
    df_original = df_original.sort_values('is_active', ascending=False)
    df_original = df_original.drop_duplicates(subset=['UGT_ID', 'substrate'], keep='first')
    df_original = df_original[['UGT_ID', 'substrate', 'UGT_Nomenclature',
       'nt_seq', 'prot_seq', 'MolecularFormula', 'ConnectivitySMILES',
       'is_active']].drop_duplicates()
    df_original['dataset'] = 'original'
    print(f"original dataset size: {len(df_original)}")
    print(df_original.columns)

    ### ESP dataset
    print("Get data from ESP...")
    df_new_ESP = pd.read_csv(os.path.join(data_dir, "data_ESP.csv"))
    print(f"ESP dataset size: {len(df_new_ESP)}")
    print(df_new_ESP.columns)


    ### EZS dataset
    print("Get data from EZS...")
    df_new_EZS = pd.read_csv(os.path.join(data_dir, "data_EZS.csv"))
    print(f"EZS dataset size: {len(df_new_EZS)}")
    print(df_new_EZS.columns)

    ### Combine datasets
    factor = 3  # augmentation factor
    # count original positive labels
    orig_pos = df_original[df_original["is_active"] == 1]
    n_target = int(factor * len(orig_pos))
    print("Original positives:", len(orig_pos))
    original_pairs = set(zip(df_original["UGT_ID"], df_original["MolecularFormula"]))
    # keep only new pairs (not in original)
    df_ESP_tmp = df_new_ESP[~df_new_ESP.apply(
        lambda row: (row["UGT_ID"], row["MolecularFormula"]) in original_pairs, axis=1
    )]
    df_EZS_tmp = df_new_EZS[~df_new_EZS.apply(
        lambda row: (row["UGT_ID"], row["MolecularFormula"]) in original_pairs, axis=1
    )]
    # take sample from ESP and EZS
    n_each = n_target // 2
    df_ESP = df_ESP_tmp.sample(n=min(n_each, len(df_ESP_tmp)), random_state=42)
    print(f"took {len(df_ESP)} (pos.) samples from ESP")
    df_EZS = df_EZS_tmp.sample(n=min(n_each, len(df_EZS_tmp)), random_state=42)
    print(f"took {len(df_EZS)} (pos.) samples from EZS")
    df_all = pd.concat([df_original, df_ESP, df_EZS], ignore_index=True)

    print(f"Merged dataset shape: {df_all.shape}\n")

    split_and_analyse_dataset(df_all, plots=True)

if __name__ == "__main__":
    main()