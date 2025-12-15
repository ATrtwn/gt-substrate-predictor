"""
Concatenate protein and substrate embeddings for Random Forest training.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

def load_embeddings(verbose=False):
    """Load all embedding files."""
    if verbose:
        print(" - Loading embeddings...")
    
    # Protein embeddings
    protein_emb = np.load(data_dir / 'Protein_Embeddings' / 'protein_embeddings_prott5.npy')
    protein_df = pd.read_csv(data_dir / 'UGT.csv')
    if verbose:
        print(f"    Protein: {protein_emb.shape} - {len(protein_df)} proteins")
    
    # Substrate embeddings
    substrate_data = {}
    
    # ChemBERTa2
    cb2_emb = np.load(data_dir / 'Substrate_Embeddings' / 'substrate_embeddings_chemberta2.npy')
    cb2_df = pd.read_csv(data_dir / 'Substrate.csv')[['substrate']]
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
    
    # KPGT
    kpgt_data = np.load(data_dir / 'Substrate' / 'kpgt_base.npz')
    kpgt_emb = kpgt_data['fps']
    kpgt_df = pd.read_csv(data_dir / 'Substrate.csv')[['substrate']]
    kpgt_df = kpgt_df.dropna(subset=['substrate']).reset_index(drop=True)
    substrate_data['kpgt'] = (kpgt_emb, kpgt_df, 2304)
    if verbose:
        print(f"    KPGT: {kpgt_emb.shape}")
    
    return protein_emb, protein_df, substrate_data


def create_concatenated_dataset(protein_emb, protein_df, substrate_emb, substrate_df, 
                                  activity_df, substrate_name, output_dir, verbose=False):
    """
    Create concatenated embeddings for protein-substrate pairs in Activity.csv.
    
    Returns:
        X: Concatenated embeddings (N, protein_dim + substrate_dim)
        y: Activity labels (N,)
        metadata: DataFrame with protein names, substrate names, etc.
    """
    if verbose:
        print(f" - Processing {substrate_name.upper()}")
    
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
        
        # Concatenate
        concat_emb = np.concatenate([p_emb, s_emb])
        
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

    if verbose:
        print(f"    Valid pairs: {len(X)}")
        print(f"    Skipped (no protein embedding): {skipped_protein}")
        print(f"    Skipped (no substrate embedding): {skipped_substrate}")
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
    protein_emb, protein_df, substrate_data = load_embeddings(verbose=verbose)
    
    # Load activity data
    activity_df = pd.read_csv(data_dir / 'Activity.csv')
    if verbose:
        print(f" - Activity.csv: {len(activity_df)} protein-substrate pairs")
    
    # Process requested substrate types
    valid_substrate_emb = ['chemberta2', 'chemberta3', 'kpgt', 'all']
    if embeddings not in valid_substrate_emb:
        raise ValueError(
            f"Invalid substrate embeddings strategy '{embeddings}'. Must be one (or multiple) of {valid_substrate_emb}."
        )
    if embeddings == 'all':
        substrate_types = ['chemberta2', 'chemberta3', 'kpgt']
    else:
        substrate_types = embeddings
    
    for sub_type in substrate_types:
        sub_emb, sub_df, sub_dim = substrate_data[sub_type]
        create_concatenated_dataset(
            protein_emb, protein_df,
            sub_emb, sub_df,
            activity_df, sub_type,
            output_dir,
            verbose=verbose
        )

    if verbose:
        print(" - All concatenations complete!")
        print(f" - Output directory: {output_dir}")
