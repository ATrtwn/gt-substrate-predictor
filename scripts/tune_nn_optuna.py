"""
Hyperparameter tuning with Optuna for GT-substrate prediction.

Usage:
    python scripts/tune_nn_optuna.py
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
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.nn_model import GT_NN, BilinearInteractionNet, AttentionMLP
from src.data.data_split import stratified_split_by_entities
from src.utils.helper_function import get_params, setup_logging
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
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


def objective(trial, config):
    """
    Optuna objective function - returns metric to MAXIMIZE.
    
    Optuna will call this function many times with different hyperparameters.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # =============================================================
    # DEFINE SEARCH SPACE - Suggest hyperparameters
    # =============================================================
    
    # Substrate embedding selection
    substrate_name = trial.suggest_categorical('substrate_name', ['chemberta2', 'chemberta3', 'kpgt'])
    
    # Model architecture - choose between MLP, Attention, or Bilinear
    model_type = trial.suggest_categorical('model_type', ['mlp', 'attention', 'bilinear'])
    
    # Hidden layer configuration (applies to all model types)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_dims = []
    for i in range(n_layers):
        dim = trial.suggest_categorical(f'hidden_dim_{i}', [128, 256, 512, 768])
        hidden_dims.append(dim)
    
    # Regularization
    dropout = trial.suggest_float('dropout', 0.2, 0.9, step=0.1)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
    
    # Training hyperparameters
    learning_rate = trial.suggest_categorical('lr', [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])  # Powers of 10: 0.000001 to 0.01
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    
    # Model-specific parameters
    if model_type == 'attention':
        num_heads = trial.suggest_categorical('num_heads', [2, 4, 8])
        use_residual = trial.suggest_categorical('use_residual', [True, False])
        projection_dim = 128
    elif model_type == 'bilinear':
        projection_dim = trial.suggest_categorical('projection_dim', [64, 128, 256])
        num_heads = 4
        use_residual = True
    else:  # mlp
        num_heads = 4
        use_residual = True
        projection_dim = 128
    
    # =============================================================
    # LOAD DATA (substrate_name is now from trial, not config)
    # =============================================================
    # =============================================================
    
    # substrate_name is now from trial suggestion, not config
    protein_name = config['protein_name']
    concatenation_path = config['concatenation_path']
    
    data_dir = Path(__file__).parent.parent
    concatenated_embeddings = np.load(data_dir / concatenation_path / f'X_{substrate_name}.npy')
    activity = np.load(data_dir / concatenation_path / f'y_{substrate_name}.npy')
    meta_name = f"metadata_{substrate_name}_{protein_name}.csv"
    metadata = pd.read_csv(data_dir / concatenation_path / meta_name)
    
    # Convert to binary
    activity_binary = (activity != "none").astype(int)
    
    # Create splits
    splits = stratified_split_by_entities(
        metadata,
        protein_col="UGT_trivial_name",
        substrate_col="substrate",
        label_col="activity",
        plot=False
    )
    
    train = splits['train']
    val = splits['val']
    
    # Get embeddings
    train_emb = concatenated_embeddings[metadata.index.isin(train.index)]
    val_emb = concatenated_embeddings[metadata.index.isin(val.index)]
    
    # Normalize
    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    val_emb = scaler.transform(val_emb)
    
    train_labels = activity_binary[metadata.index.isin(train.index)]
    val_labels = activity_binary[metadata.index.isin(val.index)]
    
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
    
    # =============================================================
    # BUILD MODEL with suggested hyperparameters
    # =============================================================
    
    input_dim = concatenated_embeddings.shape[1]
    protein_dim = 1024
    substrate_dim = input_dim - protein_dim
    
    if model_type == 'attention':
        model = AttentionMLP(
            protein_dim=protein_dim,
            substrate_dim=substrate_dim,
            num_heads=num_heads,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_residual=use_residual
        ).to(device)
    elif model_type == 'bilinear':
        model = BilinearInteractionNet(
            protein_dim=protein_dim,
            substrate_dim=substrate_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            projection_dim=projection_dim
        ).to(device)
    else:  # mlp
        # Standard MLP with flexible hidden layers
        model = GT_NN(input_dim=input_dim, hidden_dims=hidden_dims, dropout=dropout).to(device)
    
    # Loss and optimizer
    class_counts = np.bincount(train_labels)
    class_weights = torch.FloatTensor([1.0 / c for c in class_counts]).to(device)
    pos_weight = class_weights[1] / class_weights[0]
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # =============================================================
    # TRAINING LOOP with pruning
    # =============================================================
    
    max_epochs = config.get('max_epochs', 50)  # Limit epochs for faster tuning
    
    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_probs, val_true = evaluate(model, val_loader, criterion, device)
        
        # Calculate accuracy for this epoch
        val_preds = (val_probs > 0.5).astype(int)
        val_acc = accuracy_score(val_true, val_preds)
        
        # Report intermediate value to Optuna (for pruning)
        trial.report(val_acc, epoch)
        
        # Optuna pruning: Stop unpromising trials early
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    # =============================================================
    # RETURN FINAL METRIC (to maximize)
    # =============================================================
    
    # Final evaluation
    final_val_loss, final_probs, final_true = evaluate(model, val_loader, criterion, device)
    final_preds = (final_probs > 0.5).astype(int)
    
    # Calculate multiple metrics
    val_accuracy = accuracy_score(final_true, final_preds)
    val_roc_auc = roc_auc_score(final_true, final_probs)
    val_f1 = f1_score(final_true, final_preds)
    
    # Log all metrics to trial (viewable in Optuna visualization)
    trial.set_user_attr('substrate_name', substrate_name)
    trial.set_user_attr('val_accuracy', val_accuracy)
    trial.set_user_attr('val_roc_auc', val_roc_auc)
    trial.set_user_attr('val_f1', val_f1)
    trial.set_user_attr('val_loss', final_val_loss)
    
    # Return metric to MAXIMIZE (Optuna will find the best)
    return val_f1  # Optimizing for F1-score (balanced precision/recall)


def run_optuna_study(config):
    """
    Run Optuna hyperparameter optimization study.
    """
    setup_logging()
    
    logging.info("="*60)
    logging.info("Starting Optuna Hyperparameter Optimization")
    logging.info("="*60)
    logging.info(f"Substrate: {config['substrate_name']}")
    logging.info(f"Protein: {config['protein_name']}")
    logging.info(f"Number of trials: {config['n_trials']}")
    logging.info(f"Max epochs per trial: {config.get('max_epochs', 50)}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=f"gt_substrate_{config['substrate_name']}",
        direction="maximize",  # Maximize accuracy
        sampler=TPESampler(seed=42),  # Tree-structured Parzen Estimator
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)  # Less aggressive - evaluates after 10 epochs
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, config),
        n_trials=config['n_trials'],
        timeout=config.get('timeout', None),  # Optional: max time in seconds
        n_jobs=1,  # Run trials sequentially (set to -1 for parallel if you have multiple GPUs)
    )
    
    # =============================================================
    # RESULTS
    # =============================================================
    
    logging.info("="*60)
    logging.info("Optimization Complete!")
    logging.info("="*60)
    
    # Best trial
    best_trial = study.best_trial
    logging.info(f"\nBest Trial: {best_trial.number}")
    logging.info(f"Best F1-Score: {best_trial.value:.4f}")
    
    # Log substrate embedding used
    if 'substrate_name' in best_trial.user_attrs:
        logging.info(f"Best Substrate Embedding: {best_trial.user_attrs['substrate_name']}")
    
    # Log all metrics for best trial
    if 'val_accuracy' in best_trial.user_attrs:
        logging.info(f"Best Accuracy: {best_trial.user_attrs['val_accuracy']:.4f}")
    if 'val_roc_auc' in best_trial.user_attrs:
        logging.info(f"Best ROC-AUC: {best_trial.user_attrs['val_roc_auc']:.4f}")
    if 'val_loss' in best_trial.user_attrs:
        logging.info(f"Best Val Loss: {best_trial.user_attrs['val_loss']:.4f}")
    
    logging.info("\nBest Hyperparameters:")
    for key, value in best_trial.params.items():
        logging.info(f"  {key}: {value}")
    
    # Save results
    results_dir = Path(__file__).parent.parent / "reports" / "optuna"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best hyperparameters with all metrics
    best_params_path = results_dir / f"best_params_{config['substrate_name']}.json"
    with open(best_params_path, 'w') as f:
        json.dump({
            'best_f1': best_trial.value,
            'val_accuracy': best_trial.user_attrs.get('val_accuracy'),
            'val_roc_auc': best_trial.user_attrs.get('val_roc_auc'),
            'val_loss': best_trial.user_attrs.get('val_loss'),
            'best_params': best_trial.params,
            'n_trials': len(study.trials),
        }, f, indent=2)
    
    logging.info(f"\nBest parameters saved to: {best_params_path}")
    
    # Generate comprehensive optimization visualizations
    try:
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        # Create figure with 6 subplots
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Optimization history with rolling average
        ax1 = fig.add_subplot(gs[0, 0])
        trial_numbers = [t.number for t in study.trials if t.value is not None]
        trial_values = [t.value for t in study.trials if t.value is not None]
        
        ax1.scatter(trial_numbers, trial_values, alpha=0.5, s=30, label='Trials')
        ax1.plot(trial_numbers, trial_values, alpha=0.3, color='blue')
        ax1.axhline(y=best_trial.value, color='r', linestyle='--', linewidth=2, label=f'Best: {best_trial.value:.4f}')
        
        # Rolling average
        if len(trial_values) > 5:
            window = 5
            rolling_avg = [sum(trial_values[max(0, i-window):i+1]) / min(window, i+1) for i in range(len(trial_values))]
            ax1.plot(trial_numbers, rolling_avg, color='green', linewidth=2, label='Rolling Avg (5)')
        
        ax1.set_xlabel('Trial Number', fontsize=10)
        ax1.set_ylabel('F1-Score', fontsize=10)
        ax1.set_title('Optimization History', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Parameter importance
        ax2 = fig.add_subplot(gs[0, 1])
        if len(study.trials) > 5:
            try:
                importance = optuna.importance.get_param_importances(study)
                params = list(importance.keys())[:8]
                values = [importance[p] for p in params]
                
                colors = plt.cm.viridis([v/max(values) if max(values) > 0 else 0 for v in values])
                ax2.barh(params, values, color=colors)
                ax2.set_xlabel('Importance', fontsize=10)
                ax2.set_title('Hyperparameter Importance', fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3, axis='x')
            except:
                ax2.text(0.5, 0.5, 'Parameter importance\nnot available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Need >5 trials', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=10)
        
        # Plot 3: Learning rate vs Performance
        ax3 = fig.add_subplot(gs[1, 0])
        lr_values = []
        lr_scores = []
        for t in study.trials:
            if t.value is not None and 'lr' in t.params:
                lr_values.append(t.params['lr'])
                lr_scores.append(t.value)
        
        if lr_values:
            scatter = ax3.scatter(lr_values, lr_scores, c=lr_scores, cmap='RdYlGn', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            ax3.set_xscale('log')
            ax3.set_xlabel('Learning Rate', fontsize=10)
            ax3.set_ylabel('F1-Score', fontsize=10)
            ax3.set_title('Learning Rate vs Performance', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax3, label='F1-Score')
        
        # Plot 4: Dropout vs Performance
        ax4 = fig.add_subplot(gs[1, 1])
        dropout_values = []
        dropout_scores = []
        for t in study.trials:
            if t.value is not None and 'dropout' in t.params:
                dropout_values.append(t.params['dropout'])
                dropout_scores.append(t.value)
        
        if dropout_values:
            scatter = ax4.scatter(dropout_values, dropout_scores, c=dropout_scores, cmap='RdYlGn', s=80, alpha=0.7, edgecolors='black', linewidth=1)
            ax4.set_xlabel('Dropout', fontsize=10)
            ax4.set_ylabel('F1-Score', fontsize=10)
            ax4.set_title('Dropout vs Performance', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='F1-Score')
        
        # Plot 5: Model type comparison
        ax5 = fig.add_subplot(gs[2, 0])
        model_scores = {}
        for t in study.trials:
            if t.value is not None and 'model_type' in t.params:
                model = t.params['model_type']
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(t.value)
        
        if model_scores:
            models = list(model_scores.keys())
            means = [np.mean(model_scores[m]) for m in models]
            stds = [np.std(model_scores[m]) if len(model_scores[m]) > 1 else 0 for m in models]
            
            colors_map = {'mlp': 'skyblue', 'attention': 'orange', 'bilinear': 'green'}
            colors = [colors_map.get(m, 'gray') for m in models]
            
            bars = ax5.bar(models, means, yerr=stds, capsize=5, alpha=0.7, color=colors, edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax5.set_ylabel('F1-Score', fontsize=10)
            ax5.set_title('Model Architecture Comparison', fontsize=12, fontweight='bold')
            ax5.grid(True, alpha=0.3, axis='y')
        
        # Plot 6: Substrate embedding comparison
        ax6 = fig.add_subplot(gs[2, 1])
        substrate_scores = {}
        for t in study.trials:
            if t.value is not None and 'substrate_name' in t.params:
                substrate = t.params['substrate_name']
                if substrate not in substrate_scores:
                    substrate_scores[substrate] = []
                substrate_scores[substrate].append(t.value)
        
        if substrate_scores:
            substrates = list(substrate_scores.keys())
            means = [np.mean(substrate_scores[s]) for s in substrates]
            stds = [np.std(substrate_scores[s]) if len(substrate_scores[s]) > 1 else 0 for s in substrates]
            
            colors_sub = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            bars = ax6.bar(substrates, means, yerr=stds, capsize=5, alpha=0.7, 
                          color=colors_sub[:len(substrates)], edgecolor='black', linewidth=1.5)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax6.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mean:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            ax6.set_ylabel('F1-Score', fontsize=10)
            ax6.set_title('Substrate Embedding Comparison', fontsize=12, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Optuna Optimization Study - {len(study.trials)} Trials', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Save plot
        plot_path = results_dir / f"optuna_study_{config['substrate_name']}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logging.info(f"   Comprehensive visualizations saved to: {plot_path}")
        
    except Exception as e:
        logging.warning(f"Could not generate plots: {e}")
    
    return study


if __name__ == "__main__":
    # Load base config from yaml
    params = get_params('neural_network')
    nn_config = params
    
    # Optuna-specific config
    config = {
        'substrate_name': nn_config['substrate_name'],
        'protein_name': nn_config['protein_name'],
        'concatenation_path': nn_config.get('concatenation_path', 'data/concatenated_embeddings'),
        'n_trials': 1000,      
        'max_epochs': 60,
        'timeout': None,  # Optional: max time in seconds (e.g., 3600 for 1 hour)
    }
    
    # Run optimization
    study = run_optuna_study(config)
    
    logging.info("\n" + "="*60)
    logging.info("To use the best hyperparameters, update configs/neural_network.yml")
    logging.info("with the values shown above, then run: python scripts/train_nn.py")
    logging.info("="*60)
