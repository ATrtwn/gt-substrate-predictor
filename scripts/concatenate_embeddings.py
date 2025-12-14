"""
Concatenate protein and substrate embeddings for Random Forest training.

Usage:
    python scripts/concatenate_embeddings.py --substrate chemberta2
    python scripts/concatenate_embeddings.py --substrate chemberta3
    python scripts/concatenate_embeddings.py --substrate kpgt
    python scripts/concatenate_embeddings.py --substrate all  # Generate all 3
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.features.feature_utils import compute_all_features


def load_embeddings():
    """Load all embedding files."""
    ROOT = Path(__file__).parent.parent
    
    print("Loading embeddings...")
    
    # Load SMILES table
    smiles_df = pd.read_csv(
        ROOT / "data" / "Substrate_SMILES.csv"
    )[["substrate", "SMILES_isomeric_1"]]


    # Protein embeddings
    protein_emb = np.load(ROOT / 'data' / 'Protein_Embeddings' / 'protein_embeddings_prott5.npy')
    protein_df = pd.read_csv(ROOT / 'data' / 'UGT.csv')
    print(f"  Protein: {protein_emb.shape} - {len(protein_df)} proteins")
    
    # Substrate embeddings
    substrate_data = {}
    
    # ChemBERTa2
    cb2_emb = np.load(ROOT / 'data' / 'Substrate_Embeddings' / 'substrate_embeddings_chemberta2.npy')
    #cb2_df = pd.read_csv(ROOT / 'data' / 'Substrate_with_embeddings_chemberta2.csv')
    cb2_df = pd.read_csv(ROOT / 'data' / 'Substrate.csv')[['substrate']]
    substrate_data['chemberta2'] = (cb2_emb, cb2_df, 384)
    print(f"  ChemBERTa2: {cb2_emb.shape}")
    
    # ChemBERTa3
    cb3_data = torch.load(ROOT / 'data' / 'Substrate_Embeddings' / 'ChemBERTa3_substrate_embeddings.pt')
    cb3_emb = cb3_data['embeddings'].numpy()
    cb3_df = pd.DataFrame({'substrate': cb3_data['substrates']})
    substrate_data['chemberta3'] = (cb3_emb, cb3_df, 768)
    print(f"  ChemBERTa3: {cb3_emb.shape}")
    
    # KPGT
    # kpgt_data = np.load(ROOT / 'data' / 'Substrate_Embeddings' / 'kpgt.npz')
    # kpgt_emb = kpgt_data['fps']
    # kpgt_df = pd.read_csv(ROOT / 'data' / 'Substrate.csv')[['substrate']]
    # kpgt_df = kpgt_df.dropna(subset=['substrate']).reset_index(drop=True)
    # substrate_data['kpgt'] = (kpgt_emb, kpgt_df, 2304)
    # print(f"  KPGT: {kpgt_emb.shape}")
    
    return protein_emb, protein_df, substrate_data, smiles_df


def create_concatenated_dataset(protein_emb, protein_df, substrate_emb, substrate_df, 
                                  activity_df, substrate_name, output_dir, smiles_df):
    """
    Create concatenated embeddings for protein-substrate pairs in Activity.csv.
    
    Returns:
        X: Concatenated embeddings (N, protein_dim + substrate_dim)
        y: Activity labels (N,)
        metadata: DataFrame with protein names, substrate names, etc.
    """
    print(f"\n=== Processing {substrate_name.upper()} ===")
    
    # Create lookup dictionaries
    protein_to_idx = {name: idx for idx, name in enumerate(protein_df['UGT_trivial_name'])}
    substrate_to_idx = {name: idx for idx, name in enumerate(substrate_df['substrate'])}
    
    # Process each activity pair
    X_list = []
    y_list = []
    metadata_list = []
    
    skipped_protein = 0
    skipped_substrate = 0
    
    for _, row in activity_df.iterrows():
        protein_name = row['UGT_trivial_name']
        substrate_name_val = row['substrate']
        activity = row['activity']
        
        # Check if embeddings exist
        if protein_name not in protein_to_idx:
            skipped_protein += 1
            continue
        
        if substrate_name_val not in substrate_to_idx:
            skipped_substrate += 1
            continue
        
        # Get embeddings
        p_idx = protein_to_idx[protein_name]
        s_idx = substrate_to_idx[substrate_name_val]
        
        p_emb = protein_emb[p_idx]
        s_emb = substrate_emb[s_idx]

        # ===== NEW: get SMILES =====
        smiles_row = smiles_df[smiles_df["substrate"] == substrate_name_val]
        if smiles_row.empty:
            smiles = None
        else:
            smiles = smiles_row.iloc[0]["SMILES_isomeric_1"]

        # ===== NEW: get protein sequence =====
        seq = protein_df.loc[
            protein_df["UGT_trivial_name"] == protein_name,
            "prot_seq"
        ].values[0]

        # ===== NEW: compute additional features =====
        extra_feats = compute_all_features(smiles, seq)

        # clean None / nan
        extra_feats = np.array(
            [
                0.0 if (v is None or (isinstance(v, float) and np.isnan(v)))
                else float(v)
                for v in extra_feats
            ],
            dtype=float
        )

        # ===== FINAL concatenation =====
        concat_emb = np.concatenate([p_emb, s_emb, extra_feats])
        
        X_list.append(concat_emb)
        y_list.append(activity)
        metadata_list.append({
            'ID': row['ID'],
            'UGT_trivial_name': protein_name,
            'substrate': substrate_name_val,
            'activity': activity,
            'protein_idx': p_idx,
            'substrate_idx': s_idx
        })
    
    X = np.array(X_list)
    y = np.array(y_list)
    metadata = pd.DataFrame(metadata_list)
    
    print(f"  Valid pairs: {len(X)}")
    print(f"  Skipped (no protein embedding): {skipped_protein}")
    print(f"  Skipped (no substrate embedding): {skipped_substrate}")
    print(f"  Final shape: X={X.shape}, y={y.shape}")
    print(f"  Activity distribution:")
    print(f"    {pd.Series(y).value_counts().to_dict()}")
    
    # Save
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / f'X_{substrate_name}.npy', X)
    np.save(output_dir / f'y_{substrate_name}.npy', y)
    metadata.to_csv(output_dir / f'metadata_{substrate_name}.csv', index=False)
    
    print(f"  Saved to {output_dir}/")
    print(f"    - X_{substrate_name}.npy")
    print(f"    - y_{substrate_name}.npy")
    print(f"    - metadata_{substrate_name}.csv")
    
    return X, y, metadata


def main():
    parser = argparse.ArgumentParser(description='Concatenate protein and substrate embeddings')
    parser.add_argument('--substrate', type=str, default='all',
                        choices=['chemberta2', 'chemberta3', 'kpgt', 'all'],
                        help='Which substrate embedding to use (default: all)')
    parser.add_argument('--output-dir', type=str, default='data/concatenated_embeddings',
                        help='Output directory for concatenated embeddings')
    args = parser.parse_args()
    
    ROOT = Path(__file__).parent.parent
    output_dir = ROOT / args.output_dir
    
    # Load data
    protein_emb, protein_df, substrate_data, smiles_df = load_embeddings()
    
    # Load activity data
    activity_df = pd.read_csv(ROOT / 'data' / 'Activity.csv')
    print(f"\nActivity.csv: {len(activity_df)} protein-substrate pairs")
    
    # Process requested substrate types
    if args.substrate == 'all':
        substrate_types = ['chemberta2', 'chemberta3', 'kpgt']
    else:
        substrate_types = [args.substrate]
    
    for sub_type in substrate_types:
        sub_emb, sub_df, sub_dim = substrate_data[sub_type]
        create_concatenated_dataset(
            protein_emb, protein_df,
            sub_emb, sub_df,
            activity_df, sub_type,
            output_dir,
            smiles_df
        )
    
    print("\n✅ All concatenations complete!")
    print(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
