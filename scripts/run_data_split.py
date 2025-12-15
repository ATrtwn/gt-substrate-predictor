import os
import pandas as pd
from src.data.data_split import stratified_split_by_entities, check_split
from src.utils.visualization import plot_split_statistics, plot_upset_sets

from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent / "data"

def split_and_analyse_dataset(df, plots=False, verbose=False):
    """
    Dataset splitting and analysis:
        - Splits protein–substrate pairs into train, validation, and test sets
        - Enforces C1 / C2 / C3 generalization constraints
        - Computes and reports dataset statistics

    Args:
        df:
            Dataframe that should be split
        plots (bool):
            If True, plots for the graph and splits are generated
        verbose (bool):
            If True, prints progress messages for each preprocessing step
    """

    protein_col = "UGT_ID"
    substrate_col = "substrate"
    label_col = "is_active"
    if verbose:
        print(f" - Unique enzymes ({protein_col}): {df[protein_col].nunique()}")
        print(f" - Unique substrates ({substrate_col}): {df[substrate_col].nunique()}")
        print(f" - Class distribution ({label_col}):")
        print(f"{df[label_col].value_counts()}")

    df_split = stratified_split_by_entities(df,
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

    dataset_len = len(df[[protein_col, substrate_col]].drop_duplicates())
    if verbose:
        print(f" - Out of {dataset_len} distinct pairs ")
        print(" - Class distribution per split:")
    for name, subset in [("Training", train),
                         ("C1_val", c1_val), ("C2_val", c2_val), ("C3_val", c3_val),
                         ("C1_test", c1_test), ("C2_test", c2_test), ("C3_test", c3_test)]:
        counts = subset["is_active"].value_counts(normalize=True).sort_index()
        if verbose:
            print(f"   {name}: {dict(counts)} (n={len(subset)})")

    val = pd.concat([c1_val, c2_val, c3_val], axis=0)
    test = pd.concat([c1_test, c2_test, c3_test], axis=0)
    if verbose:
        print(f"   Training: {len(train)} | val: {len(val)} | test: {len(test)}")

    # check if split is valid
    check_split(train, c1_val, c2_val, c3_val, c1_test, c2_test, c3_test, protein_col, substrate_col)

    # plots
    if plots:
        plot_split_statistics(df_split, protein_col, substrate_col, label_col="is_active")
        if len(test) > 1:
            plot_upset_sets(train, val, c1_test, c2_test, c3_test)

    # save datasets
    df_split.to_csv(f"{data_dir}/split.csv", index=False)

    # you could also save the splits separately
    #train.to_csv(f"{data_dir}/train.csv", index=False)
    #c1_val.to_csv(f"{data_dir}/C1_val.csv", index=False)
    #c2_val.to_csv(f"{data_dir}/C2_val.csv", index=False)
    #c3_val.to_csv(f"{data_dir}/C3_val.csv", index=False)
    #c1_test.to_csv(f"{data_dir}/C1_test.csv", index=False)
    #c2_test.to_csv(f"{data_dir}/C2_test.csv", index=False)
    #c3_test.to_csv(f"{data_dir}/C3_test.csv", index=False)

    return df_split

if __name__ == "__main__":
    print("==== Splitting into C1/C2/C3 ====")
    df = pd.read_csv(os.path.join(data_dir, "full_dataset.csv"))
    df_all = split_and_analyse_dataset(df, plots=False, verbose=False)