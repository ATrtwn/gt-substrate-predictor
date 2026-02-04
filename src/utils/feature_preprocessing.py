import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

def feature_preprocessing(params, embeddings, metadata, activity=None, split_indices=None, fit_scaler=True, scaler=None):
    """
    Integrate and align handcrafted features with embeddings for prediction or training.
    Returns:
        - concatenated_embeddings: np.ndarray (embeddings + features if enabled)
        - aligned_activity: np.ndarray (if activity provided)
        - scaler: StandardScaler fitted on train split (if fit_scaler)
    """
    USE_HANDCRAFTED_FEATURES = params.get("use_handcrafted_features", True)
    if not USE_HANDCRAFTED_FEATURES:
        if fit_scaler:
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
        else:
            embeddings = scaler.transform(embeddings)
        if activity is not None:
            return embeddings, activity, scaler
        return embeddings, scaler

    # Load features
    features_all = np.load("data/concatenated_embeddings/features_full_dataset.npy")
    features_df = pd.read_csv("data/concatenated_embeddings/features_full_dataset.csv")
    substrate_map = pd.read_csv("data/Substrate_with_embeddings.csv", usecols=["substrate", "smiles"])
    substrate_map = substrate_map.drop_duplicates().dropna(subset=["substrate", "smiles"])
    metadata_ = metadata.copy()
    # Ensure index is a column for alignment after merge
    metadata_ = metadata_.reset_index().rename(columns={"index": "index"})
    metadata_ = pd.merge(metadata_, substrate_map, left_on="substrate", right_on="substrate", how="left")
    if 'UGT_ID' in features_df.columns:
        features_df['UGT_ID'] = features_df['UGT_ID'].astype(str)
    if 'ugt_id' in metadata_.columns:
        metadata_['ugt_id'] = metadata_['ugt_id'].astype(str)
    merged = pd.merge(
        metadata_,
        features_df,
        left_on=['ugt_id', 'smiles'],
        right_on=['UGT_ID', 'SMILES_isomeric_1'],
        how='left',
        sort=False,
        suffixes=(None, '_feat')
    )
    feature_cols = [c for c in merged.columns if c.startswith('f')]
    before_drop = merged.shape[0]
    merged = merged.dropna(subset=feature_cols)
    after_drop = merged.shape[0]
    merged = merged.reset_index(drop=True)
    features = merged[feature_cols].to_numpy(dtype=np.float32)
    kept_indices = merged['index'].values.astype(int) if 'index' in merged.columns else None
    # Align embeddings and activity robustly
    if 'index' in merged.columns:
        idx = merged['index'].values.astype(int)
        embeddings = embeddings[idx]
        if activity is not None:
            activity = activity[idx]
    else:
        raise ValueError("Merged DataFrame does not contain 'index' column for alignment. Check merge logic.")

    # Normalize features
    if fit_scaler:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    # Concatenate
    concatenated = np.concatenate([embeddings, features], axis=1)
    if activity is not None:
        return concatenated, activity, scaler, kept_indices
    return concatenated, scaler, kept_indices
