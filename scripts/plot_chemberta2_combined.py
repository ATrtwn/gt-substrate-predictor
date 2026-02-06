#!/usr/bin/env python
"""Create a single comprehensive figure showing all key ChemBERTa2 insights."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import chi2_contingency

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 12

# Paths
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "embeddings"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load training dataset
train_data = pd.read_csv(DATA_DIR / "train.csv")

# Load substrate embeddings, coordinates, and properties
emb_with_coords = pd.read_csv(EMBEDDINGS_DIR / "Substrate_with_embeddings_chemberta2.csv")
properties_df = pd.read_csv(EMBEDDINGS_DIR / "Substrate_with_properties_chemberta2.csv")
coords_2d = pd.read_csv(EMBEDDINGS_DIR / "substrate_2d_coords_tsne_chemberta2.csv")

# Add coordinates to embeddings dataframe
emb_with_coords['x'] = coords_2d['x'].values
emb_with_coords['y'] = coords_2d['y'].values

# Merge training dataset with substrate data
df = train_data[['substrate', 'is_active']].merge(
    emb_with_coords[['substrate', 'x', 'y']], 
    on='substrate', 
    how='inner'
)
df = df.merge(
    properties_df[['substrate', 'cluster_kmeans']], 
    on='substrate', 
    how='left'
)
df.rename(columns={'is_active': 'activity_mode'}, inplace=True)

# Calculate full training set statistics
full_train_active_pct = 100 * train_data['is_active'].mean()
full_train_active = train_data['is_active'].sum()
full_train_inactive = len(train_data) - full_train_active

# Calculate statistics
contingency_table = pd.crosstab(df['cluster_kmeans'], df['activity_mode'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
cluster_counts = df['cluster_kmeans'].value_counts().sort_index()
activity_counts = df['activity_mode'].value_counts()

# Calculate per-cluster activity
cluster_activity = df.groupby('cluster_kmeans')['activity_mode'].agg(['sum', 'count'])
cluster_activity['active_pct'] = 100 * cluster_activity['sum'] / cluster_activity['count']

print(f"Chi-square: {chi2:.2f}, p-value: {p_value:.2e}")

# ==============================================================================
# CREATE 2x2 COMBINED FIGURE
# ==============================================================================
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Color palettes
cluster_palette = sns.color_palette("tab10", n_colors=df['cluster_kmeans'].nunique())
activity_palette = {1: '#1f77b4', 0: '#ff7f0e'}
activity_labels = {1: 'Active', 0: 'Inactive'}

# ==============================================================================
# TOP LEFT: t-SNE colored by clusters (showing imbalance)
# ==============================================================================
ax1 = fig.add_subplot(gs[0, 0])

for i, cluster in enumerate(sorted(df['cluster_kmeans'].unique())):
    mask = df['cluster_kmeans'] == cluster
    count = mask.sum()
    ax1.scatter(
        df.loc[mask, 'x'], 
        df.loc[mask, 'y'],
        c=[cluster_palette[i]],
        label=f'C{cluster} (n={count})',
        alpha=0.7,
        s=25,
        edgecolors='white',
        linewidth=0.2
    )

ax1.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
ax1.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
ax1.set_title('A. Cluster Distribution (Imbalanced)', fontsize=14, fontweight='bold', pad=10)
legend1 = ax1.legend(
    title='Clusters',
    title_fontsize=11,
    fontsize=10,
    loc='upper right',
    frameon=True,
    ncol=1
)
legend1.get_title().set_fontweight('bold')
ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add imbalance annotation
max_size = cluster_counts.max()
min_size = cluster_counts.min()
fold_change = max_size / min_size
ax1.text(0.02, 0.98, f'{fold_change:.1f}× imbalance\n({max_size} vs {min_size})',
         transform=ax1.transAxes, ha='left', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=11, fontweight='bold')

# ==============================================================================
# TOP RIGHT: t-SNE colored by activity (showing skew)
# ==============================================================================
ax2 = fig.add_subplot(gs[0, 1])

for activity in sorted(df['activity_mode'].unique()):
    mask = df['activity_mode'] == activity
    count = mask.sum()
    # Use full training set counts for legend
    if activity == 1:
        legend_label = f'{activity_labels.get(activity, str(activity))}: {full_train_active} ({full_train_active_pct:.0f}%)'
    else:
        legend_label = f'{activity_labels.get(activity, str(activity))}: {full_train_inactive} ({100-full_train_active_pct:.0f}%)'
    
    ax2.scatter(
        df.loc[mask, 'x'], 
        df.loc[mask, 'y'],
        c=activity_palette.get(activity, '#95a5a6'),
        label=legend_label,
        alpha=0.7,
        s=25,
        edgecolors='white',
        linewidth=0.2
    )

ax2.set_xlabel('t-SNE 1', fontsize=13, fontweight='bold')
ax2.set_ylabel('t-SNE 2', fontsize=13, fontweight='bold')
ax2.set_title('B. Training Set Activity (Highly Skewed)', fontsize=14, fontweight='bold', pad=10)
legend2 = ax2.legend(
    title=f'Full Training Set (n={len(train_data)})',
    title_fontsize=11,
    fontsize=10,
    loc='upper right',
    frameon=True
)
legend2.get_title().set_fontweight('bold')
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add note about visualization
ax2.text(0.02, 0.02, f'Visualized: {len(df)} samples\nwith embeddings',
         transform=ax2.transAxes, ha='left', va='bottom',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.6),
         fontsize=9, style='italic')

# ==============================================================================
# BOTTOM LEFT: Cluster size bar chart
# ==============================================================================
ax3 = fig.add_subplot(gs[1, 0])

clusters_sorted = cluster_counts.sort_values(ascending=False)
bars = ax3.bar(
    range(len(clusters_sorted)), 
    clusters_sorted.values,
    color=[cluster_palette[i] for i in range(len(clusters_sorted))],
    edgecolor='black',
    linewidth=1,
    alpha=0.8
)

ax3.set_xlabel('Cluster ID', fontsize=13, fontweight='bold')
ax3.set_ylabel('Number of Substrates', fontsize=13, fontweight='bold')
ax3.set_title('C. Cluster Sizes (5.3× Range)', fontsize=14, fontweight='bold', pad=10)
ax3.set_xticks(range(len(clusters_sorted)))
ax3.set_xticklabels([f'{int(c)}' for c in clusters_sorted.index])
ax3.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add count labels
for bar, count in zip(bars, clusters_sorted.values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# ==============================================================================
# BOTTOM RIGHT: Activity percentages by cluster (showing correlation)
# ==============================================================================
ax4 = fig.add_subplot(gs[1, 1])

clusters = sorted(df['cluster_kmeans'].unique())
active_pcts = []
inactive_pcts = []

for cluster_id in clusters:
    active_pct = cluster_activity.loc[cluster_id, 'active_pct']
    active_pcts.append(active_pct)
    inactive_pcts.append(100 - active_pct)

x = np.arange(len(clusters))
width = 0.35

bars1 = ax4.bar(x - width/2, active_pcts, width, label='Active', 
                color='#1f77b4', edgecolor='black', linewidth=1)
bars2 = ax4.bar(x + width/2, inactive_pcts, width, label='Inactive', 
                color='#ff7f0e', edgecolor='black', linewidth=1)

# Add percentage labels
for bar, pct in zip(bars1, active_pcts):
    if pct > 5:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

for bar, pct in zip(bars2, inactive_pcts):
    if pct > 5:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.0f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_xlabel('Cluster ID', fontsize=13, fontweight='bold')
ax4.set_ylabel('Percentage (%)', fontsize=13, fontweight='bold')
ax4.set_title(f'D. Strong Cluster-Activity Correlation (χ²={chi2:.1f}, p<0.001)', 
              fontsize=14, fontweight='bold', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels([f'{int(c)}' for c in clusters])
ax4.legend(fontsize=10, loc='upper right', frameon=True)
ax4.set_ylim([0, 105])
ax4.grid(True, alpha=0.2, axis='y', linestyle='--', linewidth=0.5)
ax4.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.4)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

# Add overall title
fig.suptitle('ChemBERTa-2 Substrate Embeddings: Cluster Imbalance, Activity Skew, and Strong Correlation',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
output_path = FIGURES_DIR / "chemberta2_comprehensive_analysis.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n✓ Combined analysis plot saved to: {output_path}")

# Also save high-res version
output_path_hires = FIGURES_DIR / "chemberta2_comprehensive_analysis_hires.png"
plt.savefig(output_path_hires, dpi=600, bbox_inches='tight')
print(f"✓ High-res version saved to: {output_path_hires}")

print(f"\n{'='*70}")
print("ONE FIGURE SHOWS:")
print('='*70)
print("  A (top-left):     Cluster imbalance in embedding space")
print(f"  B (top-right):    Activity skew ({full_train_active_pct:.0f}% active in full training set)")
print(f"  C (bottom-left):  {fold_change:.1f}× cluster size variation")
print(f"  D (bottom-right): Strong correlation (χ²={chi2:.1f}, p<0.001)")
print(f"\nThis single figure makes all three key points!")
