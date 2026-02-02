"""
Ensemble prediction using multiple trained models.

Usage:
    python scripts/ensemble_predict.py --models experiments/best_model_chemberta2_seed_*.pth
    python scripts/ensemble_predict.py --models experiments/best_model_chemberta2_seed_*.pth --train-seeds

Options:
    --models         Paths to model checkpoints (can use wildcards)
    --train-seeds    Train all ensemble seeds before running ensemble (overwrites existing models)
    --substrate      Substrate embedding name (default: from config)
    --protein        Protein embedding name (default: from config)
"""

import logging
import sys
from pathlib import Path
import json
import argparse
from glob import glob
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

# Import feature_preprocessing utility
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / 'src' / 'utils'))
from feature_preprocessing import feature_preprocessing

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.nn_model import GT_NN, BilinearInteractionNet, AttentionMLP
from src.data.data_split import stratified_split_by_entities
from src.utils.helper_function import get_params, setup_logging

def load_model_from_checkpoint(checkpoint_path, model_class, **model_kwargs):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_with_model(model, data_loader, device):
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
    individual_probs = []
    for i, model_path in enumerate(model_paths, 1):
        logging.info(f"Loading model {i}/{len(model_paths)}: {model_path}")
        model = load_model_from_checkpoint(model_path, model_class, **model_kwargs)
        model.to(device)
        probs = predict_with_model(model, data_loader, device)
        individual_probs.append(probs)
    ensemble_probs = np.mean(individual_probs, axis=0)
    return ensemble_probs, individual_probs

def evaluate_ensemble(model_paths, substrate_name, protein_name, concatenation_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    params = get_params("neural_network")
    embeddings_file = Path(concatenation_path) / f"X_{substrate_name}.npy"
    metadata_file = Path(concatenation_path) / f"metadata_{substrate_name}.csv"
    embeddings = np.load(embeddings_file)
    metadata_df = pd.read_csv(metadata_file)
    full_df = pd.read_csv('data/full_dataset.csv')
    meta_cols = {col.lower(): col for col in metadata_df.columns}
    full_cols = {col.lower(): col for col in full_df.columns}
    merge_cols = []
    if 'ugt_id' in meta_cols and 'ugt_id' in full_cols:
        merge_cols.append((meta_cols['ugt_id'], full_cols['ugt_id']))
    if 'substrate' in meta_cols and 'substrate' in full_cols:
        merge_cols.append((meta_cols['substrate'], full_cols['substrate']))
    if not merge_cols:
        raise ValueError('Could not align metadata and full_dataset for activity labels.')
    left_on = [mc[0] for mc in merge_cols]
    right_on = [mc[1] for mc in merge_cols]
    dupe_mask = full_df.duplicated(subset=right_on, keep=False)
    if dupe_mask.sum() > 0:
        full_df = full_df.drop_duplicates(subset=right_on, keep='first')
    merged = pd.merge(
        metadata_df,
        full_df[['is_active'] + right_on],
        left_on=left_on,
        right_on=right_on,
        how='left',
        sort=False,
        copy=False,
        indicator=True
    )
    if merged['is_active'].isna().sum() > 0:
        raise ValueError("Some rows in metadata_df could not be matched to full_dataset.csv.")
    assert len(merged) == len(metadata_df)
    metadata_df = merged.reset_index(drop=True)
    metadata_df['label'] = metadata_df['is_active'].astype(int)
    labels = metadata_df['label'].values

    # --- Feature preprocessing (embeddings + handcrafted features) ---
    # Use train split for fitting scaler, then transform test splits
    def get_indices_from_split(split_csv, metadata):
        split_df = pd.read_csv(split_csv)
        split_cols = {col.lower(): col for col in split_df.columns}
        meta_cols = {col.lower(): col for col in metadata.columns}
        merge_cols = []
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
        return indices

    train_idx = get_indices_from_split('data/train.csv', metadata_df)
    c1_idx = get_indices_from_split('data/C1_test.csv', metadata_df)
    c2_idx = get_indices_from_split('data/C2_test.csv', metadata_df)
    c3_idx = get_indices_from_split('data/C3_test.csv', metadata_df)

    # Fit scaler on train, transform all splits
    X_train, feature_scaler, train_kept_idx = feature_preprocessing(params, embeddings, metadata_df.iloc[train_idx], activity=None, fit_scaler=True)
    X_c1, _, c1_kept_idx = feature_preprocessing(params, embeddings, metadata_df.iloc[c1_idx], activity=None, fit_scaler=False, scaler=feature_scaler)
    X_c2, _, c2_kept_idx = feature_preprocessing(params, embeddings, metadata_df.iloc[c2_idx], activity=None, fit_scaler=False, scaler=feature_scaler)
    X_c3, _, c3_kept_idx = feature_preprocessing(params, embeddings, metadata_df.iloc[c3_idx], activity=None, fit_scaler=False, scaler=feature_scaler)

    y_c1 = labels[c1_kept_idx]
    y_c2 = labels[c2_kept_idx]
    y_c3 = labels[c3_kept_idx]
    # Debug: print shapes after feature_preprocessing and before tensor conversion
    print(f"X_c1 shape: {X_c1.shape}, y_c1 shape: {y_c1.shape}")
    print(f"X_c2 shape: {X_c2.shape}, y_c2 shape: {y_c2.shape}")
    print(f"X_c3 shape: {X_c3.shape}, y_c3 shape: {y_c3.shape}")
    X_c1_tensor = torch.FloatTensor(X_c1)
    y_c1_tensor = torch.LongTensor(y_c1)
    X_c2_tensor = torch.FloatTensor(X_c2)
    y_c2_tensor = torch.LongTensor(y_c2)
    X_c3_tensor = torch.FloatTensor(X_c3)
    y_c3_tensor = torch.LongTensor(y_c3)
    c1_loader = DataLoader(TensorDataset(X_c1_tensor, y_c1_tensor), batch_size=32, shuffle=False)
    c2_loader = DataLoader(TensorDataset(X_c2_tensor, y_c2_tensor), batch_size=32, shuffle=False)
    c3_loader = DataLoader(TensorDataset(X_c3_tensor, y_c3_tensor), batch_size=32, shuffle=False)
    protein_dim = 1024
    input_dim = X_train.shape[1]
    substrate_dim = input_dim - protein_dim
    model_type = params['model_type']
    projection_dim = 128
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
    else:
        model_class = GT_NN
        model_kwargs = {
            'input_dim': input_dim,
            'hidden_dims': params['hidden_dims'],
            'dropout': params['dropout'],
            'activation': params.get('activation', 'relu')
        }
    logging.info(f"Using {model_class.__name__} with {len(model_paths)} models (input_dim={input_dim})")
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
        ensemble_preds = (ensemble_probs > 0.5).astype(int).flatten()
        acc = accuracy_score(y_true, ensemble_preds)
        f1 = f1_score(y_true, ensemble_preds)
        roc_auc = roc_auc_score(y_true, ensemble_probs)
        mcc = matthews_corrcoef(y_true, ensemble_preds)
        logging.info(f"{test_name} Ensemble - Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, MCC: {mcc:.4f}")
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
    parser.add_argument('--train-seeds', action='store_true',
                        help='If set, train all ensemble seeds before running ensemble (overwrites existing models)')
    args = parser.parse_args()
    params = get_params("neural_network")
    substrate_name = args.substrate or params['substrate_name']
    ensemble_seeds = [42, 123, 456, 789, 1337]
    if args.train_seeds:
        print("Training ensemble models for seeds:", ensemble_seeds)
        for seed in ensemble_seeds:
            model_path = f"experiments/best_model_{substrate_name}_seed_{seed}.pth"
            print(f"Training model for seed {seed} (output: {model_path}) ...")
            result = subprocess.run([
                sys.executable, "scripts/train_nn.py",
                "--seed", str(seed),
                "--save_path", model_path
            ])
            if result.returncode != 0:
                print(f"Training failed for seed {seed}, aborting ensemble.")
                sys.exit(1)
        args.models = [f"experiments/best_model_{substrate_name}_seed_{seed}.pth" for seed in ensemble_seeds]
    model_paths = []
    for pattern in args.models:
        matched = glob(pattern)
        if matched:
            model_paths.extend(matched)
        else:
            model_paths.append(pattern)
    model_paths = [Path(p) for p in model_paths if Path(p).exists()]
    if len(model_paths) == 0:
        logging.error("No valid model paths found!")
        sys.exit(1)
    logging.info(f"Found {len(model_paths)} models")
    for path in model_paths:
        logging.info(f"  - {path}")
    protein_name = args.protein or params['protein_name']
    concatenation_path = params['concatenation_path']
    results = evaluate_ensemble(model_paths, substrate_name, protein_name, concatenation_path)
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
