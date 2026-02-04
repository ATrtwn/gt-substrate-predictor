from pathlib import Path

import argparse
import os
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

data_dir = Path(__file__).resolve().parent.parent / "data"
output_path = Path(__file__).resolve().parent.parent / "boltz_input"

from ruamel.yaml import YAML

import csv
from typing import Optional, Iterable, Tuple, List


def write_boltz_yaml(sequence: str,prot_id :str, output_path: Path,msa_path:str = None):
    if msa_path== None:
        use_msa = False
    else:
        use_msa = True
    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": [str(prot_id)],
                    "sequence": sequence,
                    **({"msa": "/"+str(msa_path)} if use_msa else {})  # conditionally add msa
                }
            }
        ]
    }
 

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.dump(config, f)


# data directory
data_dir = Path(__file__).parent.parent / "data"
ACTIVITY_FILE = os.path.join(data_dir, "Activity.csv")
UGT_FILE = os.path.join(data_dir, "UGT.csv")
SUBSTRATE_FILE = os.path.join(data_dir, "Substrate.csv")


def iter_fasta(path: Path) -> Iterable[Tuple[str, str]]:
    """Simple FASTA iterator yielding (header, sequence)."""
    with open(path, "r", encoding="utf-8") as fh:
        header = None
        seq_lines: List[str] = []
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            yield header, "".join(seq_lines)



if __name__ == "__main__":
    # Load CSVs
    #df = pd.read_csv(data_path / "full_dataset.csv")

    parser = argparse.ArgumentParser(description="Generate YAML configs from a FASTA of UGT sequences.")
    parser.add_argument("--output", type=Path, required=True, help="Folder to save YAML configs")
    parser.add_argument("--msa", type=Path, required=False, help="Base path for MSA files (if not provided, no msa added)")
    parser.add_argument("--fasta", type=Path, required=True, help="Input FASTA file with UGT_ID headers")
    args = parser.parse_args()
    output_path = args.output
    msa_path = args.msa

    for header, sequence in iter_fasta(args.fasta):
        # header expected to be the UGT_ID (possibly with extra tokens) — take first token
        prot_id = header.split()[0]
        job_id = str(prot_id)
        msa_file = Path(msa_path) / f"{prot_id}.a3m" if msa_path else None
        write_boltz_yaml(sequence=sequence, prot_id=prot_id, output_path=Path(output_path) / f"config{job_id}.yaml", msa_path=msa_file)

  