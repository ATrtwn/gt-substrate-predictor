# KPGT embeddings
from multiprocessing import freeze_support
import sys
import os
from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

# Add project root to sys.path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

from third_party.kpgt.scripts.preprocess_downstream_dataset import preprocess_dataset
from third_party.kpgt.scripts.extract_features import extract_features

class Args:
    def __init__(self, data_path, dataset, path_length):
        self.data_path = data_path
        self.dataset = dataset
        self.path_length = path_length
        self.n_jobs = 32
        self.config = "base"
        self.model_path = f"{project_root}/third_party/kpgt/src/model/pretrained/base/base.pth"

# Usage
# args = Args(data_path="your/data/path", dataset="your_dataset", path_length=5)


import pandas as pd

def generate_KPGT_emb(csv_path=None, verbose=False):
    freeze_support()  # Optional, but recommended for frozen applications
    # Use full_dataset.csv by default
    if csv_path is None:
        csv_path = data_dir / "full_dataset.csv"
    # Load data
    df = pd.read_csv(csv_path)
    # Robust SMILES column detection
    if 'SMILES_isomeric_1' in df.columns:
        df['smiles'] = df['SMILES_isomeric_1']
        if verbose:
            print(f"    Using SMILES from SMILES_isomeric_1 column")
    elif 'smiles' in df.columns:
        if verbose:
            print(f"    Using existing smiles column")
    else:
        smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if smiles_cols:
            df['smiles'] = df[smiles_cols[0]]
            if verbose:
                print(f"    Using SMILES from column: {smiles_cols[0]}")
        else:
            raise ValueError("No SMILES column found. Expected 'SMILES_isomeric_1', 'smiles', or a column containing 'smiles'.")
    # Only keep rows with valid SMILES
    df = df[df['smiles'].notna() & (df['smiles'] != "")]
    # Save filtered data to a temp file for KPGT
    filtered_path = data_dir / "KPGT_input.csv"
    df.to_csv(filtered_path, index=False)
    if verbose:
        print(f"    Filtered {len(df)} valid SMILES to {filtered_path}")
    dataset = "Substrate"
    path_length = 5
    args = Args(data_path=data_dir, dataset=dataset, path_length=path_length)
    # Patch: point KPGT to the filtered file if needed (depends on downstream scripts)
    if verbose:
        print("    Starting KPGT embedding extraction...")
    preprocess_dataset(args=args)
    extract_features(args=args)
    if verbose:
        print("    Finished KPGT embedding extraction...")

# Add a main() function and __main__ block for direct execution
def main():
    generate_KPGT_emb(csv_path=None, verbose=True)

if __name__ == "__main__":
    main()