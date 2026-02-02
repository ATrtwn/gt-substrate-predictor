"""
Usage:
     python scripts/extract_features_full_dataset.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.features.feature_utils import compute_all_features
from pathlib import Path

# Paths
FULL_DATASET_PATH = Path("data/full_dataset.csv")
OUT_FEATURES_PATH = Path("data/concatenated_embeddings/features_full_dataset.npy")
OUT_FEATURES_CSV = Path("data/concatenated_embeddings/features_full_dataset.csv")

# Load full dataset
full_df = pd.read_csv(FULL_DATASET_PATH)



# Prepare feature matrix
features = []
feature_rows = []

for idx, row in tqdm(full_df.iterrows(), total=len(full_df), desc="Extracting features"):
    smiles = row["SMILES_isomeric_1"]
    ugt_id = row["UGT_ID"]
    prot_seq = row["prot_seq"] if "prot_seq" in row else ""
    # Check for invalid amino acids
    invalid_aas = set("XBUZJ")
    if any(aa in invalid_aas for aa in prot_seq):
        # Return NaN for all features if invalid
        try:
            n_feat = len(compute_all_features("C", "A"))  # dummy call to get feature length
        except Exception:
            n_feat = 0
        feat_vec = [float("nan")] * n_feat
    else:
        feat_vec = compute_all_features(smiles, prot_seq)
    features.append(feat_vec)
    feature_rows.append({"UGT_ID": ugt_id, "SMILES_isomeric_1": smiles, **{f"f{i}": v for i, v in enumerate(feat_vec)}})

features = np.array(features, dtype=np.float32)

# Save as .npy
OUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(OUT_FEATURES_PATH, features)

# Save as .csv (with IDs and SMILES)
features_df = pd.DataFrame(feature_rows)
features_df.to_csv(OUT_FEATURES_CSV, index=False)

print(f"Saved features: {features.shape} to {OUT_FEATURES_PATH} and {OUT_FEATURES_CSV}")
