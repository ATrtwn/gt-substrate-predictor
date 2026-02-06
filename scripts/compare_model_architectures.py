#!/usr/bin/env python
"""Compare performance across different model architectures."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 15

# Paths
ROOT = Path(__file__).parent.parent
METRICS_DIR = ROOT / "reports" / "metrics"
FIGURES_DIR = ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define model files to compare
# You'll need to update these filenames after running the experiments
model_files = {
    'Attention': 'nn_metrics_chemberta3_best.json',  # Already exists
    'FNN': 'nn_metrics_chemberta3_simple.json',      # To be generated
    'Bilinear': 'nn_metrics_chemberta3_bilinear.json'  # To be generated
}

print("Model Architecture Comparison")
print("="*70)

# Load metrics for all models
models_data = {}
for model_name, filename in model_files.items():
    filepath = METRICS_DIR / filename
    if filepath.exists():
        with open(filepath, 'r') as f:
            models_data[model_name] = json.load(f)
        print(f"✓ Loaded: {model_name}")
    else:
        print(f"✗ Missing: {model_name} ({filename})")

if len(models_data) < 2:
    print(f"\n⚠ Need at least 2 models for comparison. Found {len(models_data)}.")
    print("\nTo generate missing metrics:")
    print("1. Edit configs/neural_network.yml:")
    print("   - For FNN: set model_type: 'simple'")
    print("   - For Bilinear: set model_type: 'bilinear'")
    print("2. Run: python scripts/train_nn.py")
    print("3. Rename output to match filenames above")
    exit(1)

print(f"\n✓ Comparing {len(models_data)} models\n")

# ==============================================================================
# PLOT 1: Grouped bars for each test set
# ==============================================================================
fig1, axes = plt.subplots(1, 3, figsize=(18, 6))

test_sets = ['C1', 'C2', 'C3']
metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'mcc']
metric_names = {
    'accuracy': 'Accuracy',
    'f1': 'F1 Score',
    'roc_auc': 'ROC AUC',
    'mcc': 'MCC'
}

model_names = list(models_data.keys())
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for test_idx, test_set in enumerate(test_sets):
    ax = axes[test_idx]
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for model_idx, model_name in enumerate(model_names):
        data = models_data[model_name]
        values = []
        errors = []
        
        for metric in metrics_to_plot:
            val = data.get(f'{test_set}_{metric}', 0)
            err = data.get(f'{test_set}_{metric}_stderr', data.get(f'{test_set}_{metric}_se', 0))
            if err is None or (isinstance(err, float) and np.isnan(err)):
                err = 0
            values.append(val)
            errors.append(err)
        
        offset = (model_idx - len(model_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, 
                     label=model_name,
                     edgecolor='black', linewidth=1, alpha=0.85,
                     yerr=errors, capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title(f'{test_set} Test Set', fontsize=16, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_names[m] for m in metrics_to_plot], rotation=20, ha='right')
    ax.set_ylim([0.55, 1.0])
    ax.legend(fontsize=11, loc='upper right', frameon=True)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig1.suptitle('Model Architecture Comparison Across Test Sets', fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()

output1 = FIGURES_DIR / "model_architecture_comparison.png"
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"✓ Plot 1 saved: {output1}")

# ==============================================================================
# PLOT 2: F1 Score comparison across test sets
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

x = np.arange(len(test_sets))
width = 0.25

for model_idx, model_name in enumerate(model_names):
    data = models_data[model_name]
    values = []
    errors = []
    
    for test_set in test_sets:
        val = data.get(f'{test_set}_f1', 0)
        err = data.get(f'{test_set}_f1_stderr', data.get(f'{test_set}_f1_se', 0))
        if err is None or (isinstance(err, float) and np.isnan(err)):
            err = 0
        values.append(val)
        errors.append(err)
    
    offset = (model_idx - len(model_names)/2 + 0.5) * width
    bars = ax2.bar(x + offset, values, width, 
                  label=model_name,
                  edgecolor='black', linewidth=1.5, alpha=0.85,
                  yerr=errors, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels
    for bar, val, err in zip(bars, values, errors):
        height = bar.get_height()
        if height > 0.05:
            ax2.text(bar.get_x() + bar.get_width()/2., height + err + 0.01,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xlabel('Test Set', fontsize=16, fontweight='bold')
ax2.set_ylabel('F1 Score', fontsize=16, fontweight='bold')
ax2.set_title('F1 Score Comparison Across Architectures', fontsize=18, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(test_sets)
ax2.set_ylim([0.65, 0.95])
ax2.legend(fontsize=13, loc='upper right', frameon=True, fancybox=True, shadow=True)
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.tight_layout()

output2 = FIGURES_DIR / "model_f1_comparison.png"
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✓ Plot 2 saved: {output2}")

# ==============================================================================
# PLOT 3: Average performance across all test sets
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(12, 7))

x = np.arange(len(metrics_to_plot))
width = 0.25

for model_idx, model_name in enumerate(model_names):
    data = models_data[model_name]
    avg_values = []
    avg_errors = []
    
    for metric in metrics_to_plot:
        values = []
        errors = []
        for test_set in test_sets:
            val = data.get(f'{test_set}_{metric}', 0)
            err = data.get(f'{test_set}_{metric}_stderr', data.get(f'{test_set}_{metric}_se', 0))
            if err is None or (isinstance(err, float) and np.isnan(err)):
                err = 0
            values.append(val)
            errors.append(err)
        
        avg_values.append(np.mean(values))
        avg_errors.append(np.sqrt(np.sum(np.array(errors)**2)) / len(errors))
    
    offset = (model_idx - len(model_names)/2 + 0.5) * width
    bars = ax3.bar(x + offset, avg_values, width, 
                  label=model_name,
                  edgecolor='black', linewidth=1.5, alpha=0.85,
                  yerr=avg_errors, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels
    for bar, val in zip(bars, avg_values):
        height = bar.get_height()
        if height > 0.05:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xlabel('Metric', fontsize=16, fontweight='bold')
ax3.set_ylabel('Average Score', fontsize=16, fontweight='bold')
ax3.set_title('Average Performance Across All Test Sets', fontsize=18, fontweight='bold', pad=15)
ax3.set_xticks(x)
ax3.set_xticklabels([metric_names[m] for m in metrics_to_plot], rotation=15, ha='right')
ax3.set_ylim([0.60, 0.95])
ax3.legend(fontsize=13, loc='lower right', frameon=True, fancybox=True, shadow=True)
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()

output3 = FIGURES_DIR / "model_average_performance.png"
plt.savefig(output3, dpi=300, bbox_inches='tight')
print(f"✓ Plot 3 saved: {output3}")

# ==============================================================================
# Summary Table
# ==============================================================================
print(f"\n{'='*70}")
print("PERFORMANCE SUMMARY")
print('='*70)

for model_name in model_names:
    data = models_data[model_name]
    print(f"\n{model_name}:")
    for test_set in test_sets:
        f1 = data.get(f'{test_set}_f1', 0)
        acc = data.get(f'{test_set}_accuracy', 0)
        auc = data.get(f'{test_set}_roc_auc', 0)
        print(f"  {test_set}: F1={f1:.3f}, Acc={acc:.3f}, AUC={auc:.3f}")

print(f"\n{'='*70}")
print("✓ Comparison complete!")
print(f"\nGenerated {len(models_data)} model comparison:")
for name in model_names:
    print(f"  - {name}")
