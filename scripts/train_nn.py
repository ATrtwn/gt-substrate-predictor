"""
Train neural network for GT-substrate prediction.

Usage:
    python scripts/train_nn.py
"""

import logging
import sys
from pathlib import Path
import json

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


def train_epoch(model, train_loader, criterion, optimizer, device, noise_std=0.0):
    """
    Train for one epoch.
    
    Args:
        noise_std: Standard deviation for Gaussian noise augmentation (0.0 = no noise)
    """
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # Apply Gaussian noise augmentation if enabled
        if noise_std > 0:
            noise = torch.randn_like(batch_X) * noise_std
            batch_X = batch_X + noise
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.float())
        loss.backward()
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
    wandb_mode: str = "offline",
    project: str = "gt-substrate-predictor",
    concatenation_path: str = None,
):
    """
    Train neural network experiment.
    
    Args:
        data_augmentation: Enable Gaussian noise augmentation during training
        noise_std: Standard deviation of Gaussian noise (default: 0.02)
    """
    
    # Initialize W&B
    run = wandb.init(
        project=project,
        name=f"{model_type}_substrate-{substrate_name}_protein-{protein_name}_id-{nano_id()}",
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
        },
        mode=wandb_mode,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load data
    data_dir = Path(__file__).parent.parent
    concatenated_embeddings = np.load(data_dir / concatenation_path / f'X_{substrate_name}.npy')
    activity = np.load(data_dir / concatenation_path / f'y_{substrate_name}.npy')
    meta_name = f"metadata_{substrate_name}_{protein_name}.csv"
    metadata = pd.read_csv(data_dir / concatenation_path / meta_name)
    
    # Convert to binary: 0 if "none", else 1
    activity_binary = (activity != "none").astype(int)
    
    logging.info(f"Loaded {len(activity_binary)} samples")
    logging.info(f"Embedding dimension: {concatenated_embeddings.shape[1]}")
    logging.info(f"Class distribution: {np.bincount(activity_binary)}")
    
    # Create splits
    protein_col = "UGT_trivial_name"
    substrate_col = "substrate"
    label_col = "activity"
    
    logging.info("Creating data splits...")
    splits = stratified_split_by_entities(
        metadata,
        protein_col=protein_col,
        substrate_col=substrate_col,
        label_col=label_col,
        plot=False
    )
    
    train = splits['train']
    val = splits['val']
    c1 = splits['C1']
    c2 = splits['C2']
    c3 = splits['C3']
    
    # Get embeddings for each split
    train_emb = concatenated_embeddings[metadata.index.isin(train.index)]
    val_emb = concatenated_embeddings[metadata.index.isin(val.index)]
    c1_emb = concatenated_embeddings[metadata.index.isin(c1.index)]
    c2_emb = concatenated_embeddings[metadata.index.isin(c2.index)]
    
    # Normalize embeddings - fit on train, transform all
    logging.info("Normalizing embeddings with StandardScaler...")
    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    val_emb = scaler.transform(val_emb)
    c1_emb = scaler.transform(c1_emb)
    c2_emb = scaler.transform(c2_emb)
    
    train_labels = activity_binary[metadata.index.isin(train.index)]
    val_labels = activity_binary[metadata.index.isin(val.index)]
    
    logging.info(f"Train: {len(train_labels)}, Val: {len(val_labels)}")
    logging.info(f"Embeddings normalized - mean: {train_emb.mean():.4f}, std: {train_emb.std():.4f}")
    
    # Create DataLoaders
    train_dataset = TensorDataset(
        torch.FloatTensor(train_emb),
        torch.LongTensor(train_labels)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(val_emb),
        torch.LongTensor(val_labels)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = concatenated_embeddings.shape[1]
    # Determine protein and substrate dimensions
    # Assume protein is always ProtT5 (1024D), rest is substrate
    protein_dim = 1024
    substrate_dim = input_dim - protein_dim
    
    if model_type.lower() == "bilinear":
        model = BilinearInteractionNet(
            protein_dim=protein_dim, 
            substrate_dim=substrate_dim, 
            hidden_dims=hidden_dims, 
            dropout=dropout,
            projection_dim=128
        ).to(device)
        logging.info(f"Using BilinearInteractionNet with protein_dim={protein_dim}, substrate_dim={substrate_dim}")
    elif model_type.lower() == "attention":
        model = AttentionMLP(
            protein_dim=protein_dim,
            substrate_dim=substrate_dim,
            num_heads=num_heads,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_residual=use_residual
        ).to(device)
        logging.info(f"Using AttentionMLP with num_heads={num_heads}, protein_dim={protein_dim}, substrate_dim={substrate_dim}")
    else:
        model = GT_NN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
        logging.info(f"Using GT_NN with hidden_dims={hidden_dims}")
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(device)
    pos_weight = class_weights[1] / class_weights[0]
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
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
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, noise_std=augmentation_noise)
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
        # Metrics
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_preds_binary)
        val_f1 = f1_score(val_labels, val_preds_binary)
        val_roc_auc = roc_auc_score(val_labels, val_preds)
        
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
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            save_model(model, optimizer, epoch, val_loss, 
                      f"experiments/best_model_{substrate_name}.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
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
    
    for split_name, split_emb in [("C1", c1_emb), ("C2", c2_emb)]:
        split_labels = activity_binary[metadata.index.isin(splits[split_name].index)]
        
        test_dataset = TensorDataset(
            torch.FloatTensor(split_emb),
            torch.LongTensor(split_labels)
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        _, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
        test_preds_binary = (test_preds > 0.5).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(test_labels, test_preds_binary)
        f1 = f1_score(test_labels, test_preds_binary)
        roc_auc = roc_auc_score(test_labels, test_preds)
        mcc = matthews_corrcoef(test_labels, test_preds_binary)
        
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
    results_path = results_dir / f"nn_metrics_{substrate_name}_{nano_id()}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results_metrics, f, indent=2)
    
    logging.info(f"Metrics saved to {results_path}")
    
    run.finish()
    logging.info("Training complete!")


def main():
    params = get_params("neural_network")
    
    train_nn_experiment(
        substrate_name=params["substrate_name"],
        protein_name=params["protein_name"],
        model_type=params["model_type"],
        hidden_dims=params["hidden_dims"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
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
    )


if __name__ == "__main__":
    setup_logging()
    main()
