import pandas as pd
from pathlib import Path

def filter_dataset(df, min_seq_len=50, min_mol_weight=100):
    """Apply basic filtering criteria to remove incomplete or extreme data points."""
    pass

def create_fasta_original_data():
    # Locate the repository-level data directory (project_root/data/UGT.csv)
    data_dir = Path(__file__).parent.parent / "data"
    ugt_path = data_dir / "UGT.csv"

    if not ugt_path.exists():
        raise FileNotFoundError(f"UGT CSV not found at: {ugt_path.resolve()}")

    # Read CSV
    df = pd.read_csv(ugt_path)

    # Open a new FASTA file
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "UGT.fasta"

    # Create FASTA
    with output_path.open("w", encoding="utf8") as fasta_file:
        for _, row in df.iterrows():
            # prefer a human-readable name
            header = row.get("UGT_trivial_name")
            # prefer nucleotide sequence
            sequence = row.get("prot_seq")
            if pd.isna(sequence) or sequence is None or header is None:
                # skip rows without sequence or header
                continue
            fasta_file.write(f">{header}\n{sequence}\n")

    print(f"Wrote FASTA to: {output_path}")

def create_fasta_all_data():
    data_dir = Path(__file__).parent.parent.parent / "data"

    paths = {
        "UGT": data_dir / "UGT.csv",
        "ESP": data_dir / "data_ESP.csv",
        "EZS": data_dir / "data_EZS.csv",
    }

    # check files
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" +
            "\n".join(f" - {name}: {paths[name]}" for name in missing)
        )

    dfs = {}
    ugt_columns = [
        "UGT_ID",
        "UGT_trivial_name",
        "UGT_Nomenclature",
        "nt_seq",
        "prot_seq",
    ]
    for name, path in paths.items():
        df_tmp = pd.read_csv(path)
        dfs[name] = df_tmp[ugt_columns].copy()

    df = pd.concat(dfs, axis=0, ignore_index=True)

    # Open a new FASTA file
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "UGT.fasta"

    # Create FASTA
    with output_path.open("w", encoding="utf8") as fasta_file:
        for _, row in df.iterrows():
            # prefer a human-readable name
            header = row.get("UGT_trivial_name")
            # prefer nucleotide sequence
            sequence = row.get("prot_seq")
            if pd.isna(sequence) or sequence is None or header is None:
                # skip rows without sequence or header
                continue
            fasta_file.write(f">{header}\n{sequence}\n")

    print(f"Wrote FASTA to: {output_path}")

def mmseqs_clustering(fasta_path, output_dir, identity_threshold=0.9, coverage=0.8):
    """Run MMseqs2 clustering and redundancy reduction"""
    pass

def binarize_activity(df, label_col="activity"):
    """Convert multi-level activity values to binary (active/inactive)."""
    active_labels = ["low", "medium", "high", "low, high", "low, medium", "medium, high"]
    df["is_active"] = df[label_col].apply(lambda x: 1 if x in active_labels else 0)
    return df

def preprocess_pipeline(raw_data_dir, processed_dir):
    """Main preprocessing pipeline combining clustering, standardization, and splitting."""
    pass

