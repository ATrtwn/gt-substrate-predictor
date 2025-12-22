import os
import pandas as pd
from src.data.preprocessing import prepare_kpgt_data, create_fasta_file, create_original_dataset, create_full_dataset

from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent / "data"

def prepare_dataset(verbose=False):
    """
    Prepare the full dataset for downstream modeling.

    This function runs the complete preprocessing pipeline, including:
    1. Preparing data required for KPGT descriptor generation
    2. Creating FASTA files for protein sequence processing
    3. Merging all original and auxiliary data sources
    4. Constructing the final unified dataset used for modeling

    Args:
        verbose (bool): If True, prints progress messages for each preprocessing step
    """
    print("\n== [1/4] Preparing KPGT input data ==")
    prepare_kpgt_data(verbose=verbose)

    print("== [2/4] Creating FASTA file for protein sequences ==")
    create_fasta_file(verbose=verbose)

    print("== [3/4] Merging original data sources ==")
    create_original_dataset(verbose=verbose)

    print("== [4/4] Creating full merged dataset ==")
    df_all = create_full_dataset(verbose=verbose)

    return df_all


if __name__ == "__main__":
    print("==== Preprocessing dataset ====")
    df = prepare_dataset(verbose=False)
    df.to_csv(os.path.join(data_dir, "full_dataset.csv"), index=False)