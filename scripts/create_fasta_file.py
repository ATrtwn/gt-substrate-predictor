import pandas as pd
from pathlib import Path

# Locate the repository-level data directory (project_root/data/UGT.csv)
data_dir = Path(__file__).parent.parent / "data"
ugt_path = data_dir / "UGT.csv"

if not ugt_path.exists():
    raise FileNotFoundError(f"UGT CSV not found at: {ugt_path.resolve()}")

# Read CSV
df = pd.read_csv(ugt_path)

# Open a new FASTA file
output_dir = Path(__file__).parent.parent / "reports"
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
