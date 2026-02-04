"""
Train graph neural network for GT-substrate prediction.

Usage:
    python scripts/train_gnn.py
"""

import logging
import sys
from pathlib import Path
import json
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
import wandb
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.nn_model import GT_NN, BilinearInteractionNet, AttentionMLP, save_model
from src.models.gnn_models import GNNClassifier,MolecularEGNN,MolecularEGNN_Sparse
from src.data.data_split import stratified_split_by_entities, check_split
from src.utils.helper_function import get_params, setup_logging, nano_id
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler


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


def train_epoch(model, train_loader, criterion, optimizer, device, noise_std=0.0):
    """
    Train GNN for one epoch (graph classification).

    Args:
        noise_std: Standard deviation for Gaussian noise augmentation on node features (0.0 = no noise)
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)  # moves x, pos, edge_index, y, batch
        labels = batch.y

        # Optional Gaussian noise on node features
        if noise_std > 0:
            noise = torch.randn_like(batch.x) * noise_std
            batch.x = batch.x + noise

        optimizer.zero_grad()
        outputs = model(batch)  # shape: [num_graphs, num_classes]
        labels = labels.view(-1,1).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """Evaluate GNN model on validation/test set (graph classification)."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)  # moves x, pos, edge_index, y automatically if PyG batch supports it
            labels = batch.y
            labels = labels.view(-1,1).float()

            # Forward pass (graph-level classification)
            outputs = model(batch)  # shape: [num_graphs, num_classes]

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get probabilities if binary classification (num_classes=2)
            if outputs.size(1) == 1:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
    if len(data_loader)>0:
        avg_loss = total_loss / len(data_loader)
    else:
        avg_loss = 0
    if len(all_preds)>0:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
    else:
        all_preds = np.empty((0,1),dtype=np.float32)
        all_labels = np.empty((0,1),dtype=np.float32)
    return avg_loss, all_preds, all_labels


def train_gnn_experiment(
    model_type: str,
    dataset_path: str,
    hidden_dims: list,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    oversample:bool,
    weight_decay: float = 0.0,
    num_heads: int = 4,
    num_workers: int = 4,
    use_residual: bool = True,
    data_augmentation: bool = False,
    shufle_train: bool = True,
    noise_std: float = 0.02,
    wandb_mode: str = "offline",
    project: str = "gt-substrate-predictor",
    optimizer_name: str = "adam",
    scheduler_type: str = "reduce_on_plateau",
    momentum: float = 0.9,
    step_size: int = 20,
    gamma: float = 0.1,
    activation: str = "relu",
    seed: int = None,
    save_path: str = None,
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
    def load_trainset(name, filename, shuffle=False, oversample = False):
        path = Path(__file__).resolve().parent.parent.parent / dataset_path / filename
        graphs = torch.load(path, weights_only=False)
        print(f"Loaded training set with {len(graphs)} objects")
        print(f"DEBUG : Graph Keys founf : {graphs[0].keys()}")
        all_features = torch.stack([g.scalars for g in graphs])
        if oversample:
            labels = torch.tensor([int(g.y.item()) for g in graphs])
            class_counts = Counter(labels.tolist())

            if len(class_counts) == 2:
                n_neg = class_counts[0]
                n_pos = class_counts[1]

                weights = {
                    0: 1.0 / n_neg,
                    1: 1.0 / n_pos
                }
                sample_weights = torch.tensor(
                    [weights[int(y.item())] for y in labels],
                    dtype= torch.double
                )
                sampler = WeightedRandomSampler(
                    weights = sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                print(f"Oversampling enabled | Class dict | {dict(class_counts)}")
                return DataLoader(
                    graphs,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle = False,
                    num_workers = num_workers,
                    pin_memory=True
                ),all_features
            else:
                print("Oversampling skipped - dataset is not binary")
        return load_dataset(name,filename,shuffle=shuffle),all_features

    def load_dataset(name, filename, shuffle=False):
        path = Path(__file__).resolve().parent.parent.parent / dataset_path / filename
        print(f"Loading {name} from {path}")

        graphs = torch.load(path,weights_only=False)
        print(f"  -> {len(graphs)} graphs loaded")

        return DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    DATASETS = {
        "train": "train_pocket_dataset.pt",
        "val_C1": "val_C1_pocket_dataset.pt",
        "val_C2": "val_C2_pocket_dataset.pt",
        "val_C3": "val_C3_pocket_dataset.pt",
        "test_C1": "test_C1_pocket_dataset.pt",
        "test_C2": "test_C2_pocket_dataset.pt",
        "test_C3": "test_C3_pocket_dataset.pt",
    }
    # Set random seed if provided
    if seed is not None:
        set_seed(seed)
        logging.info(f"Random seed set to: {seed}")
    
    # Initialize W&B
    run_name = f"{model_type}_boltz_structure"
    if seed is not None:
        run_name += f"_seed-{seed}"
    run_name += f"_id-{nano_id()}"
    
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "model": model_type,
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
    loaders = {}
    #Create DataLoaders
    for name, filename in DATASETS.items():
        shuffle = (name == "train") and shufle_train
        if name == "train":
            loaders[name],all_features = load_trainset(name,filename,shuffle,oversample=oversample)
        else:
            loaders[name] = load_dataset(name, filename, shuffle)

    val_graphs = []

    for name in ["val_C1", "val_C2", "val_C3"]:
        path = dataset_path / DATASETS[name]
        graphs = torch.load(path,weights_only = False)
        val_graphs.extend(graphs)

    val_loader = DataLoader(
        val_graphs,
        batch_size=batch_size,
        shuffle=False,  # usually we don’t shuffle validation/test
        num_workers=num_workers,
        pin_memory=True
    )
    loaders["val"] = val_loader
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0) + 1e-8
    num_scalar_features = loaders["train"].dataset[0].scalars.shape[0]
    # Apply scaling to all graphs in train, val, test datasets
    for loader_name in DATASETS.keys():
        dataset = loaders[loader_name].dataset
        for g in dataset:
            g.scalars = (g.scalars - mean) / std
    # Initialize model
    print(f"DEBUG : Hidden dims {hidden_dims}")
    print(f"DEBUG : num scalar features {num_scalar_features}")
    if model_type in ["GATv2", "GAT", "GIN", "GraphSAGE"]:
        model = GNNClassifier(
            in_channels=loaders["train"].dataset[0].num_node_features,
            hidden_channels=hidden_dims,
            dropout=dropout,
            num_classes=1,  # Binary classification
            layer_name=model_type,
            heads=num_heads,
            use_residual=use_residual,
            scalar_dim=num_scalar_features,
            concat=True,
        ).to(device)
    elif model_type == "MolecularEGNN":
        model = MolecularEGNN_Sparse(
            in_dim=loaders["train"].dataset[0].num_node_features,
            hidden_dim=hidden_dims,
            dropout=dropout,
            num_classes=1,  
        ).to(device)
    train_labels = []
    # Calculate class weights for imbalanced data
    for batch in loaders["train"]:
        # batch.y shape: [num_graphs] for graph classification
        train_labels.append(batch.y.cpu())

    train_labels = torch.cat(train_labels).numpy().astype(int)
    # Convert to int for bincount (use original binary labels before smoothing)
    train_labels_int = np.round(train_labels).astype(int)
    if oversample:
        criterion = nn.BCEWithLogitsLoss()
    else:
        class_counts = np.bincount(train_labels_int)
        n_neg = class_counts[0]
        n_pos = class_counts[1]
        if n_pos==0:
            raise ValueError("No positive samples in training set")
        pos_weight = torch.tensor([n_neg / n_pos], device = device, dtype=torch.float32)
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
    if data_augmentation:
        logging.info(f"Data augmentation enabled: Gaussian noise with std={noise_std}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, loaders["train"], criterion, optimizer, device, noise_std=augmentation_noise)
        
        # Evaluate on VALIDATION set for early stopping (proper ML practice)
        val_loss, val_preds, val_true = evaluate(model, loaders["val"], criterion, device)
        
        # Convert smoothed labels back to binary for evaluation
        val_true_binary = np.round(val_true).astype(int)
        
        # Metrics
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
                model_path = f"experiments/best_model_{model_type}_seed_{seed}.pth"
            else:
                model_path = f"experiments/best_model_{model_type}.pth"
            save_model(model, optimizer, epoch, val_loss, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model state before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state for final evaluation")
    
    # Plot training curves
    plot_dir = Path("reports/figures")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Graph Neural Network Training - {model_type}', fontsize=14, fontweight='bold')
    
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
    plot_path = plot_dir / f"gnn_training_{model_type}_{nano_id()}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to {plot_path}")

    # ALSO save numeric learning curves (CSV and NPZ) to results directory for reproducibility ✅
    curves = {
        "train_loss": np.array(train_losses, dtype=float),
        "val_loss": np.array(val_losses, dtype=float),
        "val_accuracy": np.array(val_accuracies, dtype=float),
        "val_f1": np.array(val_f1_scores, dtype=float),
        "val_roc_auc": np.array(val_roc_aucs, dtype=float)
    }
    # Ensure results_dir exists (already created below when saving metrics, but double-check)
    results_dir.mkdir(parents=True, exist_ok=True)
    # Save NPZ
    curves_npz_path = results_dir / f"learning_curves_{model_type}_{nano_id()}.npz"
    np.savez(curves_npz_path, **curves)
    # Save CSV for easy inspection
    try:
        df_curves = pd.DataFrame({k: v for k, v in curves.items()})
        curves_csv_path = results_dir / f"learning_curves_{model_type}_{nano_id()}.csv"
        df_curves.to_csv(curves_csv_path, index=False)
    except Exception as e:
        logging.warning(f"Failed to save learning curves CSV: {e}")

    logging.info(f"Learning curves saved to {curves_npz_path}")
    
    # Final evaluation on test sets
    logging.info("Evaluating on test sets...")
    
    results_metrics = {
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
    for split_name, split_loader in [
        ("C1", loaders["test_C1"]),
        ("C2", loaders["test_C2"]),
        ("C3", loaders["test_C3"])
    ]:
        if len(loaders["test_C3"])>0:
            _, test_preds, test_labels = evaluate(model, split_loader, criterion, device)
            test_preds_binary = (test_preds > 0.5).astype(int)
        
            # Convert smoothed labels back to binary for evaluation
            test_labels_binary = np.round(test_labels).astype(int)
        
            # Calculate metrics
            acc = accuracy_score(test_labels_binary, test_preds_binary)
            f1 = f1_score(test_labels_binary, test_preds_binary)
            roc_auc = roc_auc_score(test_labels_binary, test_preds)
            mcc = matthews_corrcoef(test_labels_binary, test_preds_binary)
        
            wandb.log({
                f"{split_name}/accuracy": acc,
                f"{split_name}/f1": f1,
                f"{split_name}/roc_auc": roc_auc,
                f"{split_name}/mcc": mcc,
            })
        
            # Store test results
            results_metrics[f"{split_name}_accuracy"] = float(acc)
            results_metrics[f"{split_name}_f1"] = float(f1)
            results_metrics[f"{split_name}_roc_auc"] = float(roc_auc)
            results_metrics[f"{split_name}_mcc"] = float(mcc)
        
            logging.info(f"{split_name} - Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, MCC: {mcc:.4f}")
    
    # Save metrics to JSON
    results_dir = Path("reports/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"gnn_metrics_{model_type}_{nano_id()}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results_metrics, f, indent=2)
    
    logging.info(f"Metrics saved to {results_path}")
    
    run.finish()
    logging.info("Training complete!")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train graph neural network for GT-substrate prediction')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (for ensemble)')
    parser.add_argument('--save_path', type=str, default=None, help='Custom path to save model (for ensemble)')
    args = parser.parse_args()
    
    params = get_params("graph_neural_network")
    for model in params["model_type"]: 
        train_gnn_experiment(
            dataset_path=Path(params["dataset_path"]),
            model_type=model,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            oversample=params["oversample"],
            weight_decay=params.get("weight_decay", 0.0),
            num_heads=params.get("num_heads", 4),
            num_workers=params.get("num_workers", 4),
            use_residual=params.get("use_residual", True),
            data_augmentation=params.get("data_augmentation", False),
            shufle_train=params.get("shuffle_train", True),
            noise_std=params.get("noise_std", 0.02),
            wandb_mode=params["wandb_mode"],
            project=params["project"],
            optimizer_name=params.get("optimizer", "adam"),
            scheduler_type=params.get("scheduler", "reduce_on_plateau"),
            momentum=params.get("momentum", 0.9),
            step_size=params.get("step_size", 20),
            gamma=params.get("gamma", 0.1),
            activation=params.get("activation", "relu"),
            seed=args.seed,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    setup_logging()
    main()
