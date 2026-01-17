import pandas as pd
from pathlib import Path
data_dir = Path(__file__).parent.parent / "data"
import argparse


import os
import sys
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

data_path = Path(__file__).resolve().parent.parent / "data"
output_path = Path(__file__).resolve().parent.parent / "boltz_input" 

from ruamel.yaml import YAML


def write_boltz_yaml(sequence: str, smiles: str,prot_name :str, output_path: Path, msa_path:str = None):
    if msa_path== None:
        use_msa = False
    else:
        use_msa = True
    config = {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": prot_name,
                    "sequence": sequence,
                    **({"msa": str(msa_path)} if use_msa else {})  # conditionally add msa
                }
            },
            {
                "ligand": {
                    "id": ["B"],
                    "smiles": smiles,
                }
            },
            {
                "ligand": {
                    "id": ["C"],
                    "smiles": "C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)OP(=O)(O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)O)O",
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

   
   

if __name__ == "__main__":
    # Load CSVs
    df = pd.read_csv(data_path / "merged.csv")

    parser = argparse.ArgumentParser(description="Generate YAML configs from CSV sequences-SMILES pairs.")
    parser.add_argument("--output", type=Path, required=True, help="Folder to save YAML configs")
    parser.add_argument("--msa", type=Path, required=True, help="Base path for MSA files")
    args = parser.parse_args()

    output_path = args.output
    msa_path = args.msa
    
    
    for row in df.itertuples(index=False):
        prot_name = row.UGT_trivial_name
        sequence = row.prot_seq
        smiles = row.SMILES_isomeric_1
        msa_name = row.UGT_trivial_name
        job_id = str(row.ID)
        write_boltz_yaml(sequence = sequence,smiles = smiles, prot_name = prot_name,output_path = Path(output_path) / f"config{job_id}.yaml", msa_path=Path(msa_path) / f"{msa_name}.a3m")
    # append proteins sequence

  