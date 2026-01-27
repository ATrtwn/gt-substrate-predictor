"""
Ensemble prediction using multiple trained models.

Usage:
    python scripts/ensemble_predict.py --models experiments/best_model_chemberta2_seed_*.pth
"""

import logging
import sys
from pathlib import Path
import json
import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.nn_model import GT_NN, BilinearInteractionNet, AttentionMLP
from src.data.data_split import stratified_split_by_entities
from src.utils.helper_function import get_params, setup_logging
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


def load_model_from_checkpoint(checkpoint_path, model_class, **model_kwargs):
    """Load a model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_with_model(model, data_loader, device):
    """Get predictions from a single model."""
    model.eval()
    all_probs = []
    
    with torch.no_grad():
        for batch_X, _ in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            probs = torch.sigmoid(outputs)
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)


def ensemble_predict(model_paths, data_loader, model_class, device, **model_kwargs):
    """
    Ensemble prediction using multiple models.
    
    Args:
        model_paths: List of paths to model checkpoints
        data_loader: DataLoader for the data
        model_class: Model class to instantiate
        device: torch device
        **model_kwargs: Arguments to pass to model constructor
        
    Returns:
        ensemble_probs: Average probabilities across all models
        individual_probs: List of probabilities from each model
    """
    individual_probs = []
    
    for i, model_path in enumerate(model_paths, 1):
        logging.info(f"Loading model {i}/{len(model_paths)}: {model_path}")
        model = load_model_from_checkpoint(model_path, model_class, **model_kwargs)
        model.to(device)
        
        probs = predict_with_model(model, data_loader, device)
        individual_probs.append(probs)
    
    # Average probabilities across all models
    ensemble_probs = np.mean(individual_probs, axis=0)
    
    return ensemble_probs, individual_probs


def evaluate_ensemble(model_paths, substrate_name, protein_name, concatenation_path):
    """Evaluate ensemble on test sets."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load parameters from config
    params = get_params("neural_network")
    
    # Load data
    logging.info(f"Loading embeddings from {concatenation_path}")
    embeddings_file = Path(concatenation_path) / f"X_{substrate_name}.npy"
    metadata_file = Path(concatenation_path) / f"metadata_{substrate_name}.csv"
    
    embeddings = np.load(embeddings_file)
    metadata_df = pd.read_csv(metadata_file)
    
    logging.info(f"Loaded {len(embeddings)} samples")
    logging.info(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Convert activity to binary labels (same as train_nn.py)
    # Binary: 1 if has any activity (not "none"), 0 if no activity
    metadata_df['label'] = (metadata_df['activity'] != 'none').astype(int)
    labels = metadata_df['label'].values
    logging.info(f"Class distribution: {np.bincount(labels)}")
    
    # Add cluster_id (required by data split function, set to -1 as default)
    if 'cluster_id' not in metadata_df.columns:
        metadata_df['cluster_id'] = -1
    
    # Create data splits
    logging.info("Creating data splits...")
    df_split = stratified_split_by_entities(
        metadata_df,
        protein_col='protein_idx',
        substrate_col='substrate_idx',
        label_col='label',
        plot=False
    )
    
    train = df_split[df_split['split'] == 'train']
    val = df_split[df_split['split'].str.contains('val')]
    c1_test = df_split[df_split['split'] == 'C1_test']
    c2_test = df_split[df_split['split'] == 'C2_test']
    c3_test = df_split[df_split['split'] == 'C3_test']
    
    train_idx = train.index.tolist()
    c1_idx = c1_test.index.tolist()
    c2_idx = c2_test.index.tolist()
    c3_idx = c3_test.index.tolist()
    
    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_c1 = embeddings[c1_idx]
    y_c1 = labels[c1_idx]
    X_c2 = embeddings[c2_idx]
    y_c2 = labels[c2_idx]
    X_c3 = embeddings[c3_idx]
    y_c3 = labels[c3_idx]
    
    # Normalize embeddings
    logging.info("Normalizing embeddings with StandardScaler...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_c1 = scaler.transform(X_c1)
    X_c2 = scaler.transform(X_c2)
    X_c3 = scaler.transform(X_c3)
    
    logging.info(f"Train: {len(X_train)}, C1: {len(X_c1)}, C2: {len(X_c2)}, C3: {len(X_c3)}")
    
    # Convert to tensors
    X_c1_tensor = torch.FloatTensor(X_c1)
    y_c1_tensor = torch.LongTensor(y_c1)
    X_c2_tensor = torch.FloatTensor(X_c2)
    y_c2_tensor = torch.LongTensor(y_c2)
    X_c3_tensor = torch.FloatTensor(X_c3)
    y_c3_tensor = torch.LongTensor(y_c3)
    
    # Create data loaders
    c1_loader = DataLoader(TensorDataset(X_c1_tensor, y_c1_tensor), batch_size=32, shuffle=False)
    c2_loader = DataLoader(TensorDataset(X_c2_tensor, y_c2_tensor), batch_size=32, shuffle=False)
    c3_loader = DataLoader(TensorDataset(X_c3_tensor, y_c3_tensor), batch_size=32, shuffle=False)
    
    # Determine model class and parameters
    protein_dim = 1024  # ProtT5
    substrate_dim = embeddings.shape[1] - protein_dim
    
    model_type = params['model_type']
    # Note: Models were trained with projection_dim=128 (older default)
    projection_dim = 128  # Override config to match trained models
    
    if model_type == 'bilinear':
        model_class = BilinearInteractionNet
        model_kwargs = {
            'protein_dim': protein_dim,
            'substrate_dim': substrate_dim,
            'hidden_dims': params['hidden_dims'],
            'dropout': params['dropout'],
            'projection_dim': projection_dim,
            'activation': params.get('activation', 'relu')
        }
    elif model_type == 'attention':
        model_class = AttentionMLP
        model_kwargs = {
            'protein_dim': protein_dim,
            'substrate_dim': substrate_dim,
            'hidden_dims': params['hidden_dims'],
            'dropout': params['dropout'],
            'num_heads': params.get('num_heads', 4),
            'use_residual': params.get('use_residual', True),
            'activation': params.get('activation', 'relu')
        }
    else:  # mlp
        model_class = GT_NN
        model_kwargs = {
            'input_dim': embeddings.shape[1],
            'hidden_dims': params['hidden_dims'],
            'dropout': params['dropout'],
            'activation': params.get('activation', 'relu')
        }
    
    logging.info(f"Using {model_class.__name__} with {len(model_paths)} models")
    
    # Ensemble predictions
    results = {}
    
    for test_name, test_loader, y_true in [
        ('C1', c1_loader, y_c1),
        ('C2', c2_loader, y_c2),
        ('C3', c3_loader, y_c3)
    ]:
        logging.info(f"\nEvaluating on {test_name} test set...")
        
        ensemble_probs, individual_probs = ensemble_predict(
            model_paths, test_loader, model_class, device, **model_kwargs
        )
        
        # Ensemble predictions
        ensemble_preds = (ensemble_probs > 0.5).astype(int).flatten()
        
        # Calculate metrics
        acc = accuracy_score(y_true, ensemble_preds)
        f1 = f1_score(y_true, ensemble_preds)
        roc_auc = roc_auc_score(y_true, ensemble_probs)
        mcc = matthews_corrcoef(y_true, ensemble_preds)
        
        logging.info(f"{test_name} Ensemble - Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, MCC: {mcc:.4f}")
        
        # Calculate individual model metrics for comparison
        individual_metrics = []
        for i, probs in enumerate(individual_probs):
            preds = (probs > 0.5).astype(int).flatten()
            ind_f1 = f1_score(y_true, preds)
            individual_metrics.append(ind_f1)
            logging.info(f"  Model {i+1}: F1 = {ind_f1:.4f}")
        
        logging.info(f"  Individual F1 mean: {np.mean(individual_metrics):.4f} ± {np.std(individual_metrics):.4f}")
        logging.info(f"  Ensemble improvement: {f1 - np.mean(individual_metrics):.4f}")
        
        results[test_name] = {
            'accuracy': float(acc),
            'f1': float(f1),
            'roc_auc': float(roc_auc),
            'mcc': float(mcc),
            'individual_f1s': [float(x) for x in individual_metrics],
            'individual_mean': float(np.mean(individual_metrics)),
            'individual_std': float(np.std(individual_metrics)),
            'improvement': float(f1 - np.mean(individual_metrics))
        }
    
    # Save results
    results_dir = Path("reports/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"ensemble_metrics_{substrate_name}.json"
    
    with open(results_path, 'w') as f:
        json.dump({
            'n_models': len(model_paths),
            'model_paths': [str(p) for p in model_paths],
            'results': results
        }, f, indent=2)
    
    logging.info(f"\nEnsemble results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ensemble prediction for GT-substrate prediction')
    parser.add_argument('--models', type=str, nargs='+', required=True,
                        help='Paths to model checkpoints (can use wildcards)')
    parser.add_argument('--substrate', type=str, default=None,
                        help='Substrate embedding name (default: from config)')
    parser.add_argument('--protein', type=str, default=None,
                        help='Protein embedding name (default: from config)')
    args = parser.parse_args()
    
    # Expand wildcards in model paths
    model_paths = []
    for pattern in args.models:
        matched = glob(pattern)
        if matched:
            model_paths.extend(matched)
        else:
            model_paths.append(pattern)  # Keep as is if no match
    
    model_paths = [Path(p) for p in model_paths if Path(p).exists()]
    
    if len(model_paths) == 0:
        logging.error("No valid model paths found!")
        sys.exit(1)
    
    logging.info(f"Found {len(model_paths)} models")
    for path in model_paths:
        logging.info(f"  - {path}")
    
    # Get substrate and protein names from config if not provided
    params = get_params("neural_network")
    substrate_name = args.substrate or params['substrate_name']
    protein_name = args.protein or params['protein_name']
    concatenation_path = params['concatenation_path']
    
    # Run ensemble evaluation
    results = evaluate_ensemble(model_paths, substrate_name, protein_name, concatenation_path)
    
    # Print summary
    logging.info("\n" + "="*60)
    logging.info("ENSEMBLE SUMMARY")
    logging.info("="*60)
    logging.info(f"Number of models: {len(model_paths)}")
    for test_name in ['C1', 'C2', 'C3']:
        r = results[test_name]
        logging.info(f"\n{test_name} Test Set:")
        logging.info(f"  Ensemble F1: {r['f1']:.4f}")
        logging.info(f"  Individual mean F1: {r['individual_mean']:.4f} ± {r['individual_std']:.4f}")
        logging.info(f"  Improvement: +{r['improvement']:.4f} ({r['improvement']/r['individual_mean']*100:+.1f}%)")


if __name__ == "__main__":
    setup_logging()
    main()
