#!/usr/bin/env python
"""Create clean, digestible cluster property visualizations for the report."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = DATA_DIR / "Substrate_Embeddings"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 13

# Load data
properties_df = pd.read_csv(EMBEDDINGS_DIR / "Substrate_with_properties_chemberta2.csv")

print(f"Loaded {len(properties_df)} substrates")
print(f"Columns: {properties_df.columns.tolist()}")

# ==============================================================================
# PLOT 1: Chemical Properties by Cluster (Most Important)
# ==============================================================================
fig1, axes = plt.subplots(2, 2, figsize=(14, 10))

# Define key properties
key_props = [
    ('MW', 'Molecular Weight (Da)', 'C0'),
    ('LogP', 'LogP (Lipophilicity)', 'C1'),
    ('HBA', 'H-Bond Acceptors', 'C2'),
    ('TPSA', 'Topological Polar Surface Area (Ų)', 'C3')
]

for idx, (prop, label, ax_label) in enumerate(key_props):
    ax = axes[idx // 2, idx % 2]
    
    # Prepare data
    df_valid = properties_df[properties_df[prop].notna()].copy()
    
    if len(df_valid) > 0:
        # Create violin plot with box overlay
        parts = ax.violinplot(
            [df_valid[df_valid['cluster_kmeans'] == c][prop].values 
             for c in sorted(df_valid['cluster_kmeans'].unique())],
            positions=sorted(df_valid['cluster_kmeans'].unique()),
            showmeans=True,
            showmedians=True,
            widths=0.7
        )
        
        # Color the violins
        colors = sns.color_palette("Set2", n_colors=len(df_valid['cluster_kmeans'].unique()))
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        # Style
        ax.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontsize=14, fontweight='bold')
        ax.set_title(f'{ax_label}. {label}', fontsize=15, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(sorted(df_valid['cluster_kmeans'].unique()))
        
        # Add mean values as text
        for cluster_id in sorted(df_valid['cluster_kmeans'].unique()):
            cluster_data = df_valid[df_valid['cluster_kmeans'] == cluster_id][prop]
            mean_val = cluster_data.mean()
            ax.text(cluster_id, ax.get_ylim()[1] * 0.95, f'{mean_val:.1f}',
                   ha='center', va='top', fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

fig1.suptitle('Chemical Property Distribution Across Clusters', 
              fontsize=17, fontweight='bold', y=0.995)
plt.tight_layout()

output1 = FIGURES_DIR / "cluster_chemical_properties_clean.png"
plt.savefig(output1, dpi=300, bbox_inches='tight')
print(f"\n✓ Plot 1 saved: {output1}")

# ==============================================================================
# PLOT 2: Activity and Enzyme Interactions by Cluster
# ==============================================================================
fig2, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: Activity distribution (stacked percentage)
ax_left = axes[0]

# Calculate activity percentages
activity_data = []
clusters = sorted(properties_df['cluster_kmeans'].unique())

for cluster_id in clusters:
    cluster_df = properties_df[properties_df['cluster_kmeans'] == cluster_id]
    total = len(cluster_df)
    
    if 'activity_mode' in properties_df.columns:
        active = (cluster_df['activity_mode'] == 1).sum()
        inactive = (cluster_df['activity_mode'] == 0).sum()
    else:
        active = 0
        inactive = 0
    
    activity_data.append({
        'cluster': cluster_id,
        'active': active,
        'inactive': inactive,
        'active_pct': 100 * active / total if total > 0 else 0,
        'inactive_pct': 100 * inactive / total if total > 0 else 0,
        'total': total
    })

activity_df = pd.DataFrame(activity_data)

# Create 100% stacked bar chart
x_pos = np.arange(len(clusters))
width = 0.6

bars1 = ax_left.bar(x_pos, activity_df['active_pct'], width, 
                     label='Active', color='#1f77b4', edgecolor='black', linewidth=1.5)
bars2 = ax_left.bar(x_pos, activity_df['inactive_pct'], width, 
                     bottom=activity_df['active_pct'],
                     label='Inactive', color='#ff7f0e', edgecolor='black', linewidth=1.5)

# Add percentage labels
for i, (bar, pct) in enumerate(zip(bars1, activity_df['active_pct'])):
    if pct > 5:
        ax_left.text(i, pct/2, f'{pct:.0f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')

for i, row in activity_df.iterrows():
    if row['inactive_pct'] > 5:
        ax_left.text(i, row['active_pct'] + row['inactive_pct']/2, f"{row['inactive_pct']:.0f}%",
                    ha='center', va='center', fontsize=12, fontweight='bold', color='white')

ax_left.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
ax_left.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
ax_left.set_title('A. Activity Distribution by Cluster', fontsize=15, fontweight='bold', pad=10)
ax_left.set_xticks(x_pos)
ax_left.set_xticklabels([f'{int(c)}' for c in clusters])
ax_left.set_ylim([0, 105])
ax_left.legend(fontsize=12, loc='upper right', frameon=True)
ax_left.grid(True, alpha=0.3, axis='y')
ax_left.spines['top'].set_visible(False)
ax_left.spines['right'].set_visible(False)

# Right panel: Number of enzyme interactions
ax_right = axes[1]

if 'num_enzymes' in properties_df.columns:
    df_valid = properties_df[properties_df['num_enzymes'] > 0].copy()
    
    if len(df_valid) > 0:
        # Create box plot with individual points for outliers
        positions = sorted(df_valid['cluster_kmeans'].unique())
        data_to_plot = [df_valid[df_valid['cluster_kmeans'] == c]['num_enzymes'].values 
                       for c in positions]
        
        bp = ax_right.boxplot(data_to_plot, positions=positions, widths=0.5,
                              patch_artist=True, showfliers=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=1.5),
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(linewidth=1.5),
                              capprops=dict(linewidth=1.5),
                              flierprops=dict(marker='o', markerfacecolor='red', 
                                            markersize=4, alpha=0.5))
        
        # Color boxes by cluster
        colors = sns.color_palette("Set2", n_colors=len(positions))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        # Add mean markers
        for i, cluster_id in enumerate(positions):
            cluster_data = df_valid[df_valid['cluster_kmeans'] == cluster_id]['num_enzymes']
            mean_val = cluster_data.mean()
            ax_right.plot(cluster_id, mean_val, marker='D', color='darkblue', 
                         markersize=8, markeredgecolor='white', markeredgewidth=1, zorder=3)
        
        ax_right.set_xlabel('Cluster ID', fontsize=14, fontweight='bold')
        ax_right.set_ylabel('Number of Enzyme Interactions', fontsize=14, fontweight='bold')
        ax_right.set_title('B. Substrate Promiscuity by Cluster', fontsize=15, fontweight='bold', pad=10)
        ax_right.set_xticks(positions)
        ax_right.grid(True, alpha=0.3, axis='y')
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        
        # Add legend for mean
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='D', color='w', 
                                 markerfacecolor='darkblue', markersize=8,
                                 markeredgecolor='white', label='Mean')]
        ax_right.legend(handles=legend_elements, fontsize=12, loc='upper right')

plt.tight_layout()

output2 = FIGURES_DIR / "cluster_activity_promiscuity_clean.png"
plt.savefig(output2, dpi=300, bbox_inches='tight')
print(f"✓ Plot 2 saved: {output2}")

# ==============================================================================
# Summary Statistics
# ==============================================================================
print(f"\n{'='*70}")
print("CLUSTER PROPERTY SUMMARY")
print('='*70)

for cluster_id in sorted(properties_df['cluster_kmeans'].unique()):
    cluster_data = properties_df[properties_df['cluster_kmeans'] == cluster_id]
    n = len(cluster_data)
    
    mw_mean = cluster_data['MW'].mean()
    logp_mean = cluster_data['LogP'].mean()
    
    if 'activity_mode' in cluster_data.columns:
        active_pct = 100 * (cluster_data['activity_mode'] == 1).sum() / n
    else:
        active_pct = 0
    
    if 'num_enzymes' in cluster_data.columns:
        enzyme_mean = cluster_data[cluster_data['num_enzymes'] > 0]['num_enzymes'].mean()
    else:
        enzyme_mean = 0
    
    print(f"\nCluster {cluster_id} (n={n}):")
    print(f"  MW: {mw_mean:.0f} Da | LogP: {logp_mean:.2f}")
    print(f"  Active: {active_pct:.0f}% | Avg interactions: {enzyme_mean:.1f}")

print(f"\n{'='*70}")
print("✓ Created 2 clean, digestible visualizations!")
print(f"\n  1. {output1.name}")
print(f"     → Key chemical properties (MW, LogP, HBA, TPSA)")
print(f"\n  2. {output2.name}")
print(f"     → Activity distribution & substrate promiscuity")
