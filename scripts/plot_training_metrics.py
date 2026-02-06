#!/usr/bin/env python
"""Create clean bar plots for model training metrics."""
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

# Load metrics from JSON file
with open(METRICS_DIR / "nn_metrics_chember3_noupsample.json", 'r') as f:
    raw_metrics = json.load(f)

# Convert to expected format (stderr -> se)
metrics = {
    'model': f"{raw_metrics.get('model', 'unknown')} (No Upsampling)",
    'substrate': raw_metrics.get('substrate', 'unknown'),
    'protein': raw_metrics.get('protein', 'unknown'),
}

# Extract test set metrics
test_sets = ['C1', 'C2', 'C3']
metric_types = ['accuracy', 'f1', 'roc_auc', 'mcc']

for ts in test_sets:
    for metric in metric_types:
        metrics[f'{ts}_{metric}'] = raw_metrics.get(f'{ts}_{metric}', 0)
        # Handle stderr key and NaN values
        stderr = raw_metrics.get(f'{ts}_{metric}_stderr', 0)
        if stderr is None or (isinstance(stderr, float) and np.isnan(stderr)):
            stderr = 0
        metrics[f'{ts}_{metric}_se'] = stderr

print(f"Loaded metrics for: {metrics['model']}")
print(f"Substrate: {metrics['substrate']}")
print(f"Protein: {metrics['protein']}")

# ==============================================================================
# PLOT 1: Performance Metrics by Test Set (2x2 grid)
# ==============================================================================
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

test_sets = ['C1', 'C2', 'C3']
metrics_to_plot = ['accuracy', 'f1', 'roc_auc', 'mcc']
metric_names = {
    'accuracy': 'Accuracy',
    'f1': 'F1 Score',
    'roc_auc': 'ROC AUC',
    'mcc': 'Matthews Correlation Coefficient'
}

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx // 2, idx % 2]
    
    # Extract values and standard errors for each test set
    values = [metrics.get(f'{ts}_{metric}', 0) for ts in test_sets]
    errors = [metrics.get(f'{ts}_{metric}_se', 0) for ts in test_sets]
    
    # Create bar plot with error bars
    x_pos = np.arange(len(test_sets))
    bars = ax.bar(x_pos, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8,
                   yerr=errors, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    # Style
    ax.set_xlabel('Test Set', fontsize=16, fontweight='bold')
    ax.set_ylabel(metric_names[metric], fontsize=16, fontweight='bold')
    ax.set_title(metric_names[metric], fontsize=17, fontweight='bold', pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(test_sets)
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add horizontal line at 0.5 for reference
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

fig1.suptitle('Model Performance Across Test Sets', fontsize=19, fontweight='bold', y=0.995)
plt.tight_layout()

output1 = FIGURES_DIR / "training_metrics_by_testset_noupsample.png"
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot 1 saved: {output1}")

# ==============================================================================
# PLOT 2: Grouped Bar Chart - All Metrics Side-by-Side
# ==============================================================================
fig2, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(test_sets))
width = 0.2

# Create grouped bars with error bars
for i, metric in enumerate(metrics_to_plot):
    values = [metrics.get(f'{ts}_{metric}', 0) for ts in test_sets]
    errors = [metrics.get(f'{ts}_{metric}_se', 0) for ts in test_sets]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, values, width, 
                   label=metric_names[metric],
                   edgecolor='black', linewidth=1.2, alpha=0.85,
                   yerr=errors, capsize=3, error_kw={'linewidth': 1.5, 'ecolor': 'black'})
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        if height > 0.05:  # Only label if visible
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Test Set', fontsize=16, fontweight='bold')
ax.set_ylabel('Score', fontsize=16, fontweight='bold')
ax.set_title('Model Performance Comparison Across Test Sets', fontsize=18, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(test_sets)
ax.set_ylim([0, 1.05])
ax.legend(fontsize=14, loc='lower left', frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()

output2 = FIGURES_DIR / "training_metrics_grouped_noupsample.png"
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✓ Plot 2 saved: {output2}")

# ==============================================================================
# PLOT 3: Single Bar Chart - Average Performance
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(10, 7))

# Calculate averages and errors across test sets
avg_values = []
avg_errors = []
for metric in metrics_to_plot:
    values = [metrics.get(f'{ts}_{metric}', 0) for ts in test_sets]
    errors = [metrics.get(f'{ts}_{metric}_se', 0) for ts in test_sets]
    avg_values.append(np.mean(values))
    # Propagate error (sqrt of sum of squared errors / n)
    avg_errors.append(np.sqrt(np.sum(np.array(errors)**2)) / len(errors))

x_pos = np.arange(len(metrics_to_plot))
bars = ax3.bar(x_pos, avg_values, color='#2ca02c', edgecolor='black', linewidth=1.5, alpha=0.8,
               yerr=avg_errors, capsize=5, error_kw={'linewidth': 2, 'ecolor': 'black'})

# Add value labels
for bar, val in zip(bars, avg_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom', fontsize=14, fontweight='bold')

ax3.set_xlabel('Metric', fontsize=16, fontweight='bold')
ax3.set_ylabel('Average Score', fontsize=16, fontweight='bold')
ax3.set_title('Average Model Performance Across All Test Sets', fontsize=18, fontweight='bold', pad=15)
ax3.set_xticks(x_pos)
ax3.set_xticklabels([metric_names[m] for m in metrics_to_plot], rotation=15, ha='right')
ax3.set_ylim([0, 1.05])
ax3.grid(True, alpha=0.3, axis='y')
ax3.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

plt.tight_layout()

output3 = FIGURES_DIR / "training_metrics_average_noupsample.png"
plt.savefig(output3, dpi=300, bbox_inches='tight')
print(f"✓ Plot 3 saved: {output3}")

# ==============================================================================
# Summary Statistics
# ==============================================================================
print(f"\n{'='*70}")
print("MODEL PERFORMANCE SUMMARY")
print('='*70)
print(f"\nModel: {metrics['model']}")
print(f"  Substrate embedding: {metrics['substrate']}")
print(f"  Protein embedding: {metrics['protein']}")
print(f"  Note: No oversampling on minority class")

print(f"\nPerformance by Test Set:")
for ts in test_sets:
    print(f"\n  {ts} Test Set:")
    print(f"    Accuracy: {metrics.get(f'{ts}_accuracy', 0):.3f} ± {metrics.get(f'{ts}_accuracy_se', 0):.3f}")
    print(f"    F1 Score: {metrics.get(f'{ts}_f1', 0):.3f} ± {metrics.get(f'{ts}_f1_se', 0):.3f}")
    print(f"    ROC AUC:  {metrics.get(f'{ts}_roc_auc', 0):.3f} ± {metrics.get(f'{ts}_roc_auc_se', 0):.3f}")
    print(f"    MCC:      {metrics.get(f'{ts}_mcc', 0):.3f} ± {metrics.get(f'{ts}_mcc_se', 0):.3f}")

print(f"\nAverage Performance:")
for metric in metrics_to_plot:
    values = [metrics.get(f'{ts}_{metric}', 0) for ts in test_sets]
    errors = [metrics.get(f'{ts}_{metric}_se', 0) for ts in test_sets]
    avg = np.mean(values)
    avg_err = np.sqrt(np.sum(np.array(errors)**2)) / len(errors)
    print(f"  {metric_names[metric]:35s}: {avg:.3f} ± {avg_err:.3f}")

print(f"\n{'='*70}")
print("✓ Generated 3 bar plot visualizations!")
print(f"\n  1. {output1.name}")
print(f"     → 2x2 grid: Each metric across test sets")
print(f"\n  2. {output2.name}")
print(f"     → Grouped bars: All metrics side-by-side")
print(f"\n  3. {output3.name}")
print(f"     → Average performance across all test sets")
