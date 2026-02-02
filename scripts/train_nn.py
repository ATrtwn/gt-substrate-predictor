"""
Train neural network for GT-substrate prediction.

Usage:
    python scripts/train_nn.py
    python scripts/train_nn.py --seed 42  # For ensemble training
"""


import logging
import sys
from pathlib import Path
import json
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import wandb
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.nn_model import GT_NN, BilinearInteractionNet, AttentionMLP, save_model
from src.data.data_split import stratified_split_by_entities, check_split
from src.utils.helper_function import get_params, setup_logging, nano_id
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


def bootstrap_statistic(data, statistic_func, n_bootstrap=1000, ci_level=95):
    """
    Bootstrap any statistic with confidence intervals
    
    Parameters:
    - data: array-like of values (e.g., pident values) or list of tuples
    - statistic_func: function to compute statistic (e.g., np.mean, lambda x: np.mean(x > 80))
    - n_bootstrap: number of bootstrap samples
    - ci_level: confidence interval level
    
    Returns:
    - observed: observed statistic
    - std_error: bootstrap standard error
    - ci: confidence interval
    - bootstrap_dist: bootstrap distribution
    """
    # Convert to list to handle both arrays and lists of tuples
    if isinstance(data, np.ndarray):
        data = data.tolist()
    elif not isinstance(data, list):
        data = list(data)
    
    n = len(data)
    
    # Calculate observed statistic
    observed = statistic_func(data)
    
    # Bootstrap by resampling indices
    bootstrap_values = []
    for _ in range(n_bootstrap):
        # Resample indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        # Get bootstrap sample
        bootstrap_sample = [data[i] for i in indices]
        # Calculate statistic on bootstrap sample
        stat = statistic_func(bootstrap_sample)
        bootstrap_values.append(stat)
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Standard error = standard deviation of bootstrap distribution
    std_error = np.std(bootstrap_values, ddof=1)
    
    # Confidence interval
    lower = np.percentile(bootstrap_values, (100 - ci_level) / 2)
    upper = np.percentile(bootstrap_values, 100 - (100 - ci_level) / 2)
    
    return {
        'observed': observed,
        'std_error': std_error,
        'ci': (lower, upper),
        'bootstrap_dist': bootstrap_values
    }


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, train_loader, criterion, optimizer, device, noise_std=0.0, grad_clip=None, mixup_alpha=0.0):
    """
    Train for one epoch.
    
    Args:
        noise_std: Standard deviation for Gaussian noise augmentation (0.0 = no noise)
        grad_clip: Gradient clipping value (None = no clipping)
        mixup_alpha: Mixup alpha parameter (0.0 = no mixup)
    """
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Apply Mixup augmentation if enabled
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            batch_size = batch_X.size(0)
            index = torch.randperm(batch_size).to(device)
            batch_X = lam * batch_X + (1 - lam) * batch_X[index]
            batch_y = lam * batch_y + (1 - lam) * batch_y[index]
        
        # Apply Gaussian noise augmentation if enabled
        if noise_std > 0:
            noise = torch.randn_like(batch_X) * noise_std
            batch_X = batch_X + noise
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.float())
        loss.backward()
        
        # Apply gradient clipping if enabled
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.float())
            total_loss += loss.item()
            
            # Get probabilities
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    return avg_loss, np.array(all_preds), np.array(all_labels)


def train_nn_experiment(
    substrate_name: str,
    protein_name: str,
    model_type: str,
    hidden_dims: list,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    weight_decay: float = 0.0,
    num_heads: int = 4,
    use_residual: bool = True,
    data_augmentation: bool = False,
    noise_std: float = 0.02,
    grad_clip: float = None,
    mixup_alpha: float = 0.0,
    stochastic_depth: float = 0.0,
    wandb_mode: str = "offline",
    project: str = "gt-substrate-predictor",
    concatenation_path: str = None,
    optimizer_name: str = "adam",
    scheduler_type: str = "reduce_on_plateau",
    momentum: float = 0.9,
    step_size: int = 20,
    gamma: float = 0.1,
    projection_dim: int = 128,
    activation: str = "relu",
    label_smoothing: float = 0.0,
    seed: int = None,
    save_path: str = None,
    params: dict = None,
):
    """
    Train neural network experiment.
    
    Args:
        data_augmentation: Enable Gaussian noise augmentation during training
        noise_std: Standard deviation of Gaussian noise (default: 0.02)
        label_smoothing: Label smoothing factor (0.0 = disabled, 0.1-0.2 recommended)
        seed: Random seed for reproducibility (for ensemble training)
        save_path: Custom path to save the model (for ensemble training)
    """
    
    # Set random seed if provided
    if seed is not None:
        set_seed(seed)
        logging.info(f"Random seed set to: {seed}")
    
    # Initialize W&B
    run_name = f"{model_type}_substrate-{substrate_name}_protein-{protein_name}"
    if seed is not None:
        run_name += f"_seed-{seed}"
    run_name += f"_id-{nano_id()}"
    
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "model": model_type,
            "substrate": substrate_name,
            "protein": protein_name,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "activation": activation,
        },
        mode=wandb_mode,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data and metadata
    concatenated_embeddings = np.load(f"{concatenation_path}/X_{substrate_name}.npy")
    meta_name = f"metadata_{substrate_name}.csv"
    metadata = pd.read_csv(f"{concatenation_path}/{meta_name}")
    
    # Ensure metadata and embeddings have matching sizes from the start
    if len(metadata) != len(concatenated_embeddings):
        logging.warning(f"Metadata size ({len(metadata)}) doesn't match embeddings size ({len(concatenated_embeddings)}). Truncating to embeddings size.")
        metadata = metadata.iloc[:len(concatenated_embeddings)].copy()
    
    # Add original index column to track rows through merges
    metadata['_original_idx'] = np.arange(len(metadata))
    
    # Load full_dataset.csv and use is_active for activity labels
    full_df = pd.read_csv("data/full_dataset.csv")
    # Ensure merge columns are string for robust matching
    if 'UGT_ID' in full_df.columns:
        full_df['UGT_ID'] = full_df['UGT_ID'].astype(str)
    if 'ugt_id' in metadata.columns:
        metadata['ugt_id'] = metadata['ugt_id'].astype(str)
    # Merge metadata with full_df to get is_active for each row
    metadata = pd.merge(metadata, full_df[['UGT_ID', 'substrate', 'is_active']], left_on=['ugt_id', 'substrate'], right_on=['UGT_ID', 'substrate'], how='left', suffixes=(None, '_full'))
    
    # Filter out rows with missing is_active (unmatched in full_dataset.csv)
    before_filter = len(metadata)
    valid_rows = metadata['is_active'].notna()
    metadata = metadata[valid_rows].copy()
    after_filter = len(metadata)
    if after_filter < before_filter:
        logging.warning(f"Dropped {before_filter - after_filter} samples with missing is_active labels")
    
    # Filter embeddings using original indices
    valid_indices = metadata['_original_idx'].to_numpy()
    concatenated_embeddings = concatenated_embeddings[valid_indices]
    
    # Reset index and remove tracking column
    metadata = metadata.drop(columns=['_original_idx']).reset_index(drop=True)
    
    activity = metadata['is_active'].to_numpy()

    # === HANDCRAFTED FEATURE INTEGRATION (toggleable via config) ===
    if params is None:
        raise ValueError("params dictionary must be provided to train_nn_experiment for feature toggling.")
    USE_HANDCRAFTED_FEATURES = params.get("use_handcrafted_features", True)
    if USE_HANDCRAFTED_FEATURES:
        features_all = np.load("data/concatenated_embeddings/features_full_dataset.npy")
        features_df = pd.read_csv("data/concatenated_embeddings/features_full_dataset.csv")
        # Build mapping from substrate name to SMILES
        substrate_map = pd.read_csv("data/Substrate_with_embeddings.csv", usecols=["substrate", "smiles"])
        substrate_map = substrate_map.drop_duplicates().dropna(subset=["substrate", "smiles"])
        # Merge metadata with substrate_map to get SMILES for each row
        metadata_ = metadata.copy()
        metadata_ = pd.merge(metadata_, substrate_map, left_on="substrate", right_on="substrate", how="left")
        # Now merge with features on (ugt_id, SMILES)
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
        # Drop samples with missing features (any NaN in feature columns)
        before_drop = merged.shape[0]
        merged = merged.dropna(subset=feature_cols)
        after_drop = merged.shape[0]
        if after_drop < before_drop:
            logging.warning(f"Dropped {before_drop - after_drop} samples with missing features after merge.")
        features = merged[feature_cols].to_numpy(dtype=np.float32)
        features_all_aligned = features
        # Store original indices before resetting
        original_indices = merged.index.to_numpy()
        # Filter embeddings and activity using original indices
        concatenated_embeddings = concatenated_embeddings[original_indices]
        # Now reset metadata index and update activity
        metadata = merged.reset_index(drop=True)
        activity = metadata['is_active'].to_numpy()  # Always use aligned is_active
        logging.info(f"Features loaded and aligned: shape = {features_all_aligned.shape}")
    # === END HANDCRAFTED FEATURE INTEGRATION ===


    # Print unique values for debugging
    # print("Unique activity values:", np.unique(activity, return_counts=True))

    # Auto-detect binarization
    if activity.dtype.kind in {'U', 'S', 'O'}:
        # String or object: treat 'none' as negative, else positive
        activity_binary = (activity != "none").astype(int)
    else:
        # Numeric: assume already binarized (0/1)
        activity_binary = activity.astype(int)

    # Apply label smoothing if enabled
    if label_smoothing > 0:
        logging.info(f"Applying label smoothing: epsilon={label_smoothing}")
        activity_binary = activity_binary.astype(float)
        activity_binary[activity_binary == 0] = label_smoothing  # 0 → epsilon
        activity_binary[activity_binary == 1] = 1 - label_smoothing  # 1 → 1-epsilon

    logging.info(f"Loaded {len(activity_binary)} samples")
    logging.info(f"Embedding dimension: {concatenated_embeddings.shape[1]}")
    logging.info(f"Class distribution: {np.bincount(activity_binary.astype(int))}")

    # Load precomputed splits from CSVs
    train = pd.read_csv("data/train.csv")
    val1 = pd.read_csv("data/C1_val.csv")
    val2 = pd.read_csv("data/C2_val.csv")
    val3 = pd.read_csv("data/C3_val.csv")
    val = pd.concat([val1, val2, val3], ignore_index=True)
    c1_test = pd.read_csv("data/C1_test.csv") if Path("data/C1_test.csv").exists() else None
    c2_test = pd.read_csv("data/C2_test.csv") if Path("data/C2_test.csv").exists() else None
    c3_test = pd.read_csv("data/C3_test.csv") if Path("data/C3_test.csv").exists() else None

    def get_embeddings_for_split(split_df, metadata, concatenated_embeddings):
        split_cols = {col.lower(): col for col in split_df.columns}
        meta_cols = {col.lower(): col for col in metadata.columns}
        merge_cols = []
        # Ensure UGT_ID/ugt_id columns are both strings for merge
        for col in ['UGT_ID', 'ugt_id']:
            if col in split_df.columns:
                split_df[col] = split_df[col].astype(str)
            if col in metadata.columns:
                metadata[col] = metadata[col].astype(str)
        if 'ugt_id' in meta_cols and ('ugt_id' in split_cols or 'ugt_id' in [c.lower() for c in split_df.columns]):
            merge_cols.append(('UGT_ID' if 'UGT_ID' in split_df.columns else split_cols.get('ugt_id', 'ugt_id'), meta_cols['ugt_id']))
        elif 'ugt_id' in meta_cols and 'UGT_ID' in split_cols:
            merge_cols.append(('UGT_ID', meta_cols['ugt_id']))
        if 'substrate' in meta_cols and 'substrate' in split_cols:
            merge_cols.append(('substrate', 'substrate'))
        if merge_cols:
            left_on = [mc[0] for mc in merge_cols]
            right_on = [mc[1] for mc in merge_cols]
            merged = pd.merge(split_df, metadata.reset_index(), left_on=left_on, right_on=right_on, how='inner')
            indices = merged['index'].values.astype(int)
        else:
            indices = metadata.index.isin(split_df.index).nonzero()[0]
        # Also return indices for feature normalization
        return concatenated_embeddings[indices], activity_binary[indices], indices

    # --- Print class distribution for each split (removed for production) ---
    split_info = [
        ("train", train),
        ("C1_val", val1),
        ("C2_val", val2),
        ("C3_val", val3),
        ("C1_test", c1_test),
        ("C2_test", c2_test),
        ("C3_test", c3_test),
    ]
    for name, df in split_info:
        if df is not None:
            _, labels, _ = get_embeddings_for_split(df, metadata, concatenated_embeddings)
            # print(f"{name} class distribution:", np.bincount(labels.astype(int)))

    # Split leakage check removed for production

    train_emb, train_labels, train_idx = get_embeddings_for_split(train, metadata, concatenated_embeddings)
    val_emb, val_labels, val_idx = get_embeddings_for_split(val, metadata, concatenated_embeddings)
    c1_emb, c1_labels, c1_idx = get_embeddings_for_split(c1_test, metadata, concatenated_embeddings) if c1_test is not None else (None, None, None)
    c2_emb, c2_labels, c2_idx = get_embeddings_for_split(c2_test, metadata, concatenated_embeddings) if c2_test is not None else (None, None, None)
    c3_emb, c3_labels, c3_idx = get_embeddings_for_split(c3_test, metadata, concatenated_embeddings) if c3_test is not None else (None, None, None)

    # --- Normalize features if enabled ---
    if USE_HANDCRAFTED_FEATURES:
        feature_dim = features_all_aligned.shape[1]
        # Get features for each split using indices
        train_features = features_all_aligned[train_idx]
        val_features = features_all_aligned[val_idx]
        c1_features = features_all_aligned[c1_idx] if c1_idx is not None else None
        c2_features = features_all_aligned[c2_idx] if c2_idx is not None else None
        c3_features = features_all_aligned[c3_idx] if c3_idx is not None else None

        # Fit scaler on train, transform all
        feature_scaler = StandardScaler()
        train_features = feature_scaler.fit_transform(train_features)
        val_features = feature_scaler.transform(val_features)
        if c1_features is not None:
            c1_features = feature_scaler.transform(c1_features)
        if c2_features is not None:
            c2_features = feature_scaler.transform(c2_features)
        if c3_features is not None:
            c3_features = feature_scaler.transform(c3_features)

        # Concatenate normalized features to embeddings for each split
        train_emb = np.concatenate([train_emb, train_features], axis=1)
        val_emb = np.concatenate([val_emb, val_features], axis=1)
        if c1_emb is not None:
            c1_emb = np.concatenate([c1_emb, c1_features], axis=1)
        if c2_emb is not None:
            c2_emb = np.concatenate([c2_emb, c2_features], axis=1)
        if c3_emb is not None:
            c3_emb = np.concatenate([c3_emb, c3_features], axis=1)
        logging.info("Handcrafted features normalized and concatenated to embeddings for all splits.")

    # Normalize embeddings - fit on train, transform all
    logging.info("Normalizing embeddings with StandardScaler...")
    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    val_emb = scaler.transform(val_emb)
    if c1_emb is not None:
        c1_emb = scaler.transform(c1_emb)
    if c2_emb is not None:
        c2_emb = scaler.transform(c2_emb)
    if c3_emb is not None:
        c3_emb = scaler.transform(c3_emb)

    logging.info(f"Train: {len(train_labels)}, Val: {len(val_labels)}, C1: {len(c1_labels) if c1_labels is not None else 0}, C2: {len(c2_labels) if c2_labels is not None else 0}, C3: {len(c3_labels) if c3_labels is not None else 0}")
    logging.info(f"Embeddings normalized - mean: {train_emb.mean():.4f}, std: {train_emb.std():.4f}")

    # --- Oversample minority class in training set (configurable) ---
    if params.get("oversample", True):
        from collections import Counter
        rng = np.random.default_rng(seed)
        class_counts = Counter(np.round(train_labels).astype(int))
        min_class = min(class_counts, key=class_counts.get)
        max_class = max(class_counts, key=class_counts.get)
        n_to_add = class_counts[max_class] - class_counts[min_class]
        if n_to_add > 0:
            min_indices = np.where(np.round(train_labels).astype(int) == min_class)[0]
            add_indices = rng.choice(min_indices, size=n_to_add, replace=True)
            train_emb = np.concatenate([train_emb, train_emb[add_indices]], axis=0)
            train_labels = np.concatenate([train_labels, train_labels[add_indices]], axis=0)
            logging.info(f"Oversampled minority class {min_class}: added {n_to_add} samples. New train shape: {train_emb.shape}")
        else:
            logging.info("No oversampling needed: classes already balanced.")
    else:
        logging.info("Oversampling disabled via config.")
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_emb),
        torch.FloatTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_emb),
        torch.FloatTensor(val_labels)
    )
    c1_dataset = TensorDataset(
        torch.FloatTensor(c1_emb),
        torch.FloatTensor(c1_labels)
    )
    c2_dataset = TensorDataset(
        torch.FloatTensor(c2_emb),
        torch.FloatTensor(c2_labels)
    )
    c3_dataset = TensorDataset(
        torch.FloatTensor(c3_emb),
        torch.FloatTensor(c3_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    c1_loader = DataLoader(c1_dataset, batch_size=batch_size, shuffle=False)
    c2_loader = DataLoader(c2_dataset, batch_size=batch_size, shuffle=False)
    c3_loader = DataLoader(c3_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    protein_dim = 1024
    # Recalculate input_dim and substrate_dim after all preprocessing (including oversampling/features)
    input_dim = train_emb.shape[1]
    substrate_dim = input_dim - protein_dim
    logging.info(f"Model input_dim={input_dim}, protein_dim={protein_dim}, substrate_dim={substrate_dim}")
    if USE_HANDCRAFTED_FEATURES:
        feature_dim = features.shape[1]
        logging.info(f"Handcrafted features detected: feature_dim={feature_dim}, substrate_dim (embedding+features)={substrate_dim}")
    else:
        logging.info(f"No handcrafted features: substrate_dim={substrate_dim}")
    
    if model_type.lower() == "bilinear":
        model = BilinearInteractionNet(
            protein_dim=protein_dim, 
            substrate_dim=substrate_dim, 
            hidden_dims=hidden_dims, 
            dropout=dropout,
            projection_dim=128,
            activation=activation,
            stochastic_depth=stochastic_depth
        ).to(device)
        logging.info(f"Using BilinearInteractionNet with protein_dim={protein_dim}, substrate_dim={substrate_dim}")
    elif model_type.lower() == "attention":
        model = AttentionMLP(
            protein_dim=protein_dim,
            substrate_dim=substrate_dim,
            num_heads=num_heads,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_residual=use_residual,
            activation=activation,
            stochastic_depth=stochastic_depth
        ).to(device)
        logging.info(f"Using AttentionMLP with num_heads={num_heads}, protein_dim={protein_dim}, substrate_dim={substrate_dim}")
    else:
        model = GT_NN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout, activation=activation, stochastic_depth=stochastic_depth).to(device)
        logging.info(f"Using GT_NN with hidden_dims={hidden_dims}")
    
    # Calculate class weights for imbalanced data
    # Convert to int for bincount (use original binary labels before smoothing)
    train_labels_int = np.round(train_labels).astype(int)
    class_counts = np.bincount(train_labels_int)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(device)
    pos_weight = class_weights[1] / class_weights[0]
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create optimizer based on config
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create learning rate scheduler
    if scheduler_type.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    elif scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    logging.info(f"Optimizer: {optimizer_name}")
    logging.info(f"Scheduler: {scheduler_type}")
    logging.info(f"Weight decay (L2 reg): {weight_decay}")
    
    logging.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_roc_aucs = []
    
    # Determine noise level for augmentation
    augmentation_noise = noise_std if data_augmentation else 0.0
    if grad_clip is not None:
        logging.info(f"Gradient clipping enabled: max_norm={grad_clip}")
    if stochastic_depth > 0:
        logging.info(f"Stochastic depth enabled: drop_prob={stochastic_depth}")
    if mixup_alpha > 0:
        logging.info(f"Mixup augmentation enabled: alpha={mixup_alpha}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, 
                                noise_std=augmentation_noise, grad_clip=grad_clip, mixup_alpha=mixup_alpha)

        # Evaluate on VALIDATION set for early stopping (proper ML practice)
        val_loss, val_preds, val_true = evaluate(model, val_loader, criterion, device)

        # Convert smoothed labels back to binary for evaluation
        val_true_binary = np.round(val_true).astype(int)

        # Metrics at default threshold 0.5
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_acc = accuracy_score(val_true_binary, val_preds_binary)
        val_f1 = f1_score(val_true_binary, val_preds_binary)
        val_roc_auc = roc_auc_score(val_true_binary, val_preds)

        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_roc_aucs.append(val_roc_auc)

        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_roc_auc": val_roc_auc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type.lower() == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:  # step or cosine
                scheduler.step()

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # Save best model (with seed suffix if provided)
            if save_path is not None:
                model_path = save_path
            elif seed is not None:
                model_path = f"experiments/best_model_{substrate_name}_seed_{seed}.pth"
            else:
                model_path = f"experiments/best_model_{substrate_name}.pth"
            save_model(model, optimizer, epoch, val_loss, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break

    best_threshold = 0.5  # Using default threshold
    
    # Load best model state before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state for final evaluation")
    
    # Plot training curves
    plot_dir = Path("reports/figures")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Neural Network Training - {substrate_name}', fontsize=14, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(val_f1_scores, label='Val F1', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1, 1].plot(val_roc_aucs, label='Val ROC-AUC', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('Validation ROC-AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plot_dir / f"nn_training_{substrate_name}_{nano_id()}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to {plot_path}")
    
    # Final evaluation on test sets
    logging.info("Evaluating on test sets...")
    
    results_metrics = {
        "substrate": substrate_name,
        "protein": protein_name,
        "model": model_type,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_trained": len(train_losses),
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
    }
    
    # Evaluate on all three test sets: C1, C2, and C3
    for split_name, split_loader, split_labels in [
        ("C1", c1_loader, c1_labels),
        ("C2", c2_loader, c2_labels),
        ("C3", c3_loader, c3_labels)
    ]:
        _, test_preds, test_labels = evaluate(model, split_loader, criterion, device)
        test_preds_binary = (test_preds > best_threshold).astype(int)
        test_labels_binary = np.round(test_labels).astype(int)

        # Calculate metrics
        acc = accuracy_score(test_labels_binary, test_preds_binary)
        f1 = f1_score(test_labels_binary, test_preds_binary)
        roc_auc = roc_auc_score(test_labels_binary, test_preds)
        mcc = matthews_corrcoef(test_labels_binary, test_preds_binary)

        # Compute standard errors using bootstrap_statistic
        n_bootstrap = 1000
        acc_boot = bootstrap_statistic(
            list(zip(test_labels_binary, test_preds_binary)),
            lambda arr: accuracy_score([x[0] for x in arr], [x[1] for x in arr]),
            n_bootstrap=n_bootstrap
        )
        f1_boot = bootstrap_statistic(
            list(zip(test_labels_binary, test_preds_binary)),
            lambda arr: f1_score([x[0] for x in arr], [x[1] for x in arr]),
            n_bootstrap=n_bootstrap
        )
        roc_auc_boot = bootstrap_statistic(
            list(zip(test_labels_binary, test_preds)),
            lambda arr: roc_auc_score([x[0] for x in arr], [x[1] for x in arr]),
            n_bootstrap=n_bootstrap
        )
        mcc_boot = bootstrap_statistic(
            list(zip(test_labels_binary, test_preds_binary)),
            lambda arr: matthews_corrcoef([x[0] for x in arr], [x[1] for x in arr]),
            n_bootstrap=n_bootstrap
        )

        wandb.log({
            f"{split_name}/accuracy": acc,
            f"{split_name}/f1": f1,
            f"{split_name}/roc_auc": roc_auc,
            f"{split_name}/mcc": mcc,
            f"{split_name}/accuracy_stderr": acc_boot['std_error'],
            f"{split_name}/f1_stderr": f1_boot['std_error'],
            f"{split_name}/roc_auc_stderr": roc_auc_boot['std_error'],
            f"{split_name}/mcc_stderr": mcc_boot['std_error'],
        })

        # Store test results and standard errors
        results_metrics[f"{split_name}_accuracy"] = float(acc)
        results_metrics[f"{split_name}_accuracy_stderr"] = float(acc_boot['std_error'])
        results_metrics[f"{split_name}_f1"] = float(f1)
        results_metrics[f"{split_name}_f1_stderr"] = float(f1_boot['std_error'])
        results_metrics[f"{split_name}_roc_auc"] = float(roc_auc)
        results_metrics[f"{split_name}_roc_auc_stderr"] = float(roc_auc_boot['std_error'])
        results_metrics[f"{split_name}_mcc"] = float(mcc)
        results_metrics[f"{split_name}_mcc_stderr"] = float(mcc_boot['std_error'])

        logging.info(f"{split_name} - Acc: {acc:.4f} (SE={acc_boot['std_error']:.4f}), F1: {f1:.4f} (SE={f1_boot['std_error']:.4f}), ROC-AUC: {roc_auc:.4f} (SE={roc_auc_boot['std_error']:.4f}), MCC: {mcc:.4f} (SE={mcc_boot['std_error']:.4f}) (threshold={best_threshold:.2f})")
    
    # Save metrics to JSON
    results_dir = Path("reports/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"nn_metrics_{substrate_name}_{nano_id()}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results_metrics, f, indent=2)
    
    logging.info(f"Metrics saved to {results_path}")
    
    run.finish()
    logging.info("Training complete!")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train neural network for GT-substrate prediction')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (for ensemble)')
    parser.add_argument('--save_path', type=str, default=None, help='Custom path to save model (for ensemble)')
    args = parser.parse_args()
    
    params = get_params("neural_network")
    
    train_nn_experiment(
        substrate_name=params["substrate_name"],
        protein_name=params["protein_name"],
        model_type=params["model_type"],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        grad_clip=params.get("grad_clip", None),
        mixup_alpha=params.get("mixup_alpha", 0.0),
        stochastic_depth=params.get("stochastic_depth", 0.0),
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        weight_decay=params.get("weight_decay", 0.0),
        num_heads=params.get("num_heads", 4),
        use_residual=params.get("use_residual", True),
        data_augmentation=params.get("data_augmentation", False),
        noise_std=params.get("noise_std", 0.02),
        wandb_mode=params["wandb_mode"],
        project=params["project"],
        concatenation_path=params["concatenation_path"],
        optimizer_name=params.get("optimizer", "adam"),
        scheduler_type=params.get("scheduler", "reduce_on_plateau"),
        momentum=params.get("momentum", 0.9),
        step_size=params.get("step_size", 20),
        gamma=params.get("gamma", 0.1),
        projection_dim=params.get("projection_dim", 128),
        activation=params.get("activation", "relu"),
        label_smoothing=params.get("label_smoothing", 0.0),
        seed=args.seed,
        save_path=args.save_path,
        params=params,
    )


if __name__ == "__main__":
    setup_logging()
    main()
