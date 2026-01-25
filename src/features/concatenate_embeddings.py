"""
Concatenate protein and substrate embeddings for Random Forest training.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from src.features.feature_utils import compute_all_features

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

def load_embeddings(verbose=False):
    """Load all embedding files."""
    if verbose:
        print(" - Loading embeddings...")
    

    # Try to use full_dataset.csv if available, else fallback to old files
    import os
    full_dataset_path = data_dir / "full_dataset.csv"
    substrate_data = {}
    # ChemBERTa2
    cb2_emb = np.load(data_dir / 'Substrate_Embeddings' / 'substrate_embeddings_chemberta2.npy')
    # Use the saved substrate list for perfect alignment
    cb2_list_path = data_dir / 'Substrate_Embeddings' / 'substrate_list_chemberta2.csv'
    if cb2_list_path.exists():
        cb2_df = pd.read_csv(cb2_list_path)
        if verbose:
            print(f"    ChemBERTa2: {cb2_emb.shape} (using substrate_list_chemberta2.csv: {len(cb2_df)})")
    else:
        cb2_df = pd.read_csv(data_dir / 'full_dataset.csv')[['substrate']]
        if verbose:
            print(f"    ChemBERTa2: {cb2_emb.shape} (WARNING: substrate_list_chemberta2.csv not found, using full_dataset.csv)")
    substrate_data['chemberta2'] = (cb2_emb, cb2_df, 384)
    # ChemBERTa3
    cb3_data = torch.load(data_dir / 'Substrate_Embeddings' / 'ChemBERTa3_substrate_embeddings.pt')
    cb3_emb = cb3_data['embeddings'].numpy()
    # Use the saved substrate list for perfect alignment
    cb3_df = pd.DataFrame({'substrate': cb3_data['substrates']})
    substrate_data['chemberta3'] = (cb3_emb, cb3_df, 768)
    if verbose:
        print(f"    ChemBERTa3: {cb3_emb.shape} (using ChemBERTa3_substrate_embeddings.pt: {len(cb3_df)})")

    if full_dataset_path.exists():
        df = pd.read_csv(full_dataset_path)
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]

        # Find substrate and SMILES columns
        substrate_col = next((c for c in df.columns if 'substrate' in c), None)
        smiles_col = next((c for c in df.columns if 'smiles_isomeric_1' in c or 'smiles' in c), None)
        if not substrate_col:
            raise ValueError("No substrate column found in full_dataset.csv.")
        if not smiles_col:
            raise ValueError("No SMILES column found in full_dataset.csv.")
        smiles_df = df[[substrate_col, smiles_col]].rename(columns={substrate_col: 'substrate', smiles_col: 'SMILES_isomeric_1'})

        # Protein info
        protein_emb = np.load(data_dir / 'Protein_Embeddings' / 'protein_embeddings_prott5.npy')
        # Load the mapping CSV for robust lookup
        mapping_path = data_dir / 'Protein_Embeddings' / 'protein_embedding_mapping.csv'
        if not mapping_path.exists():
            raise FileNotFoundError(f"Protein embedding mapping file not found: {mapping_path}")
        protein_df = pd.read_csv(mapping_path)
        # Normalize column names for downstream code
        protein_df.columns = [c.strip() for c in protein_df.columns]
        protein_df = protein_df.rename(columns={
            'UGT_ID': 'UGT_ID',
            'UGT_Nomenclature': 'UGT_trivial_name',
            'prot_seq': 'prot_seq',
            'ugt_id': 'UGT_ID',
            'ugt_nomenclature': 'UGT_trivial_name'
        })
        if verbose:
            print(f"    Protein: {protein_emb.shape} - {len(protein_df)} proteins (from mapping CSV)")

        # Instead of loading Activity.csv, return the full_dataset DataFrame for use as activity_df
        activity_df = df.copy()
        return protein_emb, protein_df, substrate_data, smiles_df, activity_df
    else:
        # Load SMILES table
        smiles_df = pd.read_csv(
            data_dir / "Substrate_SMILES.csv"
        )[["substrate", "SMILES_isomeric_1"]]

        # Protein embeddings
        protein_emb = np.load(data_dir / 'Protein_Embeddings' / 'protein_embeddings_prott5.npy')
        protein_df = pd.read_csv(data_dir / 'full_dataset.csv')
        if verbose:
            print(f"    Protein: {protein_emb.shape} - {len(protein_df)} proteins")
    
    # Substrate embeddings
    substrate_data = {}
    
    # ChemBERTa2
    cb2_emb = np.load(data_dir / 'Substrate_Embeddings' / 'substrate_embeddings_chemberta2.npy')
    cb2_df = pd.read_csv(data_dir / 'full_dataset.csv')[['substrate']]
    substrate_data['chemberta2'] = (cb2_emb, cb2_df, 384)
    if verbose:
        print(f"    ChemBERTa2: {cb2_emb.shape}")
    
    # ChemBERTa3
    cb3_data = torch.load(data_dir / 'Substrate_Embeddings' / 'ChemBERTa3_substrate_embeddings.pt')
    cb3_emb = cb3_data['embeddings'].numpy()
    cb3_df = pd.DataFrame({'substrate': cb3_data['substrates']})
    substrate_data['chemberta3'] = (cb3_emb, cb3_df, 768)
    if verbose:
        print(f"    ChemBERTa3: {cb3_emb.shape}")
    
    # KPGT (disabled)
    # kpgt_data = np.load(data_dir / 'Substrate' / 'kpgt_base.npz')
    # kpgt_emb = kpgt_data['fps']
    # kpgt_df = pd.read_csv(data_dir / 'Substrate.csv')[['substrate']]
    # kpgt_df = kpgt_df.dropna(subset=['substrate']).reset_index(drop=True)
    # substrate_data['kpgt'] = (kpgt_emb, kpgt_df, 2304)
    # if verbose:
    #     print(f"    KPGT: {kpgt_emb.shape}")
    
    return protein_emb, protein_df, substrate_data, smiles_df, None


def create_concatenated_dataset(protein_emb, protein_df, substrate_emb, substrate_df, 
                                  activity_df, substrate_name, output_dir, smiles_df, verbose=False):
    """
    Create concatenated embeddings for protein-substrate pairs in Activity.csv.
    
    Returns:
        X: Concatenated embeddings (N, protein_dim + substrate_dim)
        y: Activity labels (N,)
        metadata: DataFrame with protein names, substrate names, etc.
    """
    if verbose:
        print(f" - Processing {substrate_name.upper()}")
    
    # Always use UGT_ID for protein matching (robust for new dataset)
    protein_to_idx_id = {str(name).lower(): idx for idx, name in enumerate(protein_df['UGT_ID'])}
    substrate_to_idx = {name: idx for idx, name in enumerate(substrate_df['substrate'])}
    
    # Process each activity pair
    X_list = []
    y_list = []
    metadata_list = []
    
    skipped_protein = 0
    skipped_substrate = 0
    skipped_protein_list = []
    skipped_substrate_list = []
    


    # Always use 'is_active' as the label column
    activity_columns = [c.lower() for c in activity_df.columns]
    protein_id_col = 'ugt_id' if 'ugt_id' in activity_columns else 'id'
    substrate_col = 'substrate'
    activity_col = 'is_active'

    for _, row in activity_df.iterrows():
        protein_name_id = str(row[protein_id_col]).lower()
        substrate_name_val = row[substrate_col]
        activity = row[activity_col]
        # Defensive: convert to int if possible
        if pd.isna(activity):
            activity = 0
        try:
            activity = int(activity)
        except Exception:
            activity = 0

        # Only match by UGT_ID
        if protein_name_id in protein_to_idx_id:
            p_idx = protein_to_idx_id[protein_name_id]
            seqs = protein_df.loc[
                protein_df["UGT_ID"].astype(str).str.lower() == protein_name_id,
                "prot_seq"
            ].values
        else:
            print(f"[DEBUG] Protein not found by UGT_ID: {row.get(protein_id_col, 'NA')}")
            skipped_protein += 1
            skipped_protein_list.append(f"{row.get(protein_id_col, 'NA')}")
            continue

        if substrate_name_val not in substrate_to_idx:
            skipped_substrate += 1
            skipped_substrate_list.append(substrate_name_val)
            continue

        s_idx = substrate_to_idx[substrate_name_val]
        p_emb = protein_emb[p_idx]
        s_emb = substrate_emb[s_idx]

        # get SMILES
        smiles_row = smiles_df[smiles_df["substrate"] == substrate_name_val]
        if smiles_row.empty:
            smiles = None
        else:
            smiles = smiles_row.iloc[0]["SMILES_isomeric_1"]

        # get protein sequence
        if len(seqs) == 0:
            print(f"[DEBUG] Protein sequence not found for: {row.get(protein_name_trivial_col, 'NA')} / {row.get(protein_id_col, 'NA')}")
            skipped_protein += 1
            skipped_protein_list.append(f"{row.get(protein_name_trivial_col, 'NA')} / {row.get(protein_id_col, 'NA')}")
            continue
        seq = seqs[0]
        # Replace 'X' with 'A' to avoid Biopython errors
        seq = seq.replace('X', 'A')

        # compute additional features
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

        # concatenation
        concat_emb = np.concatenate([p_emb, s_emb, extra_feats])

        X_list.append(concat_emb)
        y_list.append(activity)
        # Use the matched UGT_ID for metadata
        matched_name = row.get(protein_id_col, None)
        metadata_list.append({
            'ID': row.get('id', row.get('ID', None)),
            'ugt_id': matched_name,
            'substrate': substrate_name_val,
            'activity': activity,
            'protein_idx': p_idx,
            'substrate_idx': s_idx
        })
    
    X = np.array(X_list)
    y = np.array(y_list)
    metadata = pd.DataFrame(metadata_list)

    print(f"    Valid pairs: {len(X)}")
    print(f"    Skipped (no protein embedding): {skipped_protein}")
    if skipped_protein > 0:
        print(f"    Skipped protein IDs/names (first 10): {skipped_protein_list[:10]}")
    print(f"    Skipped (no substrate embedding): {skipped_substrate}")
    if skipped_substrate > 0:
        print(f"    Skipped substrate names (first 10): {skipped_substrate_list[:10]}")
    print(f"    Final shape: X={X.shape}, y={y.shape}")
    print(f"    Activity distribution:")
    print(f"    {pd.Series(y).value_counts().to_dict()}")
    
    # Save
    output_dir.mkdir(exist_ok=True, parents=True)
    np.save(output_dir / f'X_{substrate_name}.npy', X)
    np.save(output_dir / f'y_{substrate_name}.npy', y)
    metadata.to_csv(output_dir / f'metadata_{substrate_name}.csv', index=False)

    if verbose:
        print(f"    -> Saved to {output_dir}/")
        print(f"      - X_{substrate_name}.npy")
        print(f"      - y_{substrate_name}.npy")
        print(f"      - metadata_{substrate_name}.csv")
    
    return X, y, metadata


def concatenate_embeddings(
    embeddings = 'all',
    output_dir: str = None,
    verbose = False
):
    """
        Concatenate protein and substrate embeddings into datasets.

        Args:
            embeddings: Which substrate embedding to use ('chemberta2', 'chemberta3', 'kpgt', or 'all').
            output_dir: Where to save the concatenated embeddings.
        """

    if output_dir is None:
        output_dir = data_dir/ 'concatenated_embeddings'
    
    # Load data
    protein_emb, protein_df, substrate_data, smiles_df, activity_df = load_embeddings(verbose=verbose)
    # If activity_df is None, fall back to Activity.csv (legacy mode)
    if activity_df is None:
        activity_df = pd.read_csv(data_dir / 'full_dataset.csv')
        if verbose:
            print(f" - full_dataset.csv: {len(activity_df)} protein-substrate pairs")
    else:
        if verbose:
            print(f" - Using full_dataset.csv for protein-substrate pairs: {len(activity_df)} rows")
    
    # Process requested substrate types
    valid_substrate_emb = ['chemberta2', 'chemberta3', 'all']
    if embeddings not in valid_substrate_emb:
        raise ValueError(
            f"Invalid substrate embeddings strategy '{embeddings}'. Must be one (or multiple) of {valid_substrate_emb}."
        )
    if embeddings == 'all':
        substrate_types = ['chemberta2', 'chemberta3']
    else:
        substrate_types = embeddings
    
    for sub_type in substrate_types:
        sub_emb, sub_df, sub_dim = substrate_data[sub_type]
        create_concatenated_dataset(
            protein_emb, protein_df,
            sub_emb, sub_df,
            activity_df, sub_type,
            output_dir,
            smiles_df,
            verbose=verbose
        )

    if verbose:
        print(" - All concatenations complete!")
        print(f" - Output directory: {output_dir}")
