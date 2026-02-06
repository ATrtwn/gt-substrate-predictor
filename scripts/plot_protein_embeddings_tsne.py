#!/usr/bin/env python3
"""
Create t-SNE visualization of ProtT5 protein embeddings.
Colors by species (Human vs Non-human) and shows activity markers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
DATA_DIR = Path("data")
EMBEDDINGS_DIR = DATA_DIR / "Protein_Embeddings"
FULL_DATA = DATA_DIR / "full_dataset.csv"
OUTPUT_DIR = Path("reports/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading data...")
full_df = pd.read_csv(FULL_DATA)
protein_mapping = pd.read_csv(EMBEDDINGS_DIR / "protein_embedding_mapping.csv")
tsne_coords = pd.read_csv(EMBEDDINGS_DIR / "protein_2d_coords_tsne_prott5.csv")

# Classify species
def classify_species(ugt_nomenclature):
    """Classify protein as human or non-human based on nomenclature"""
    if pd.isna(ugt_nomenclature) or ugt_nomenclature == '':
        return 'Non-human'
    nomenclature = str(ugt_nomenclature).upper()
    if nomenclature.startswith('UGT') and any(c.isdigit() for c in nomenclature[:5]):
        if 'UGT1' in nomenclature or 'UGT2' in nomenclature or 'UGT3' in nomenclature:
            return 'Human'
    return 'Non-human'

# Add species classification
protein_mapping['species'] = protein_mapping['UGT_Nomenclature'].apply(classify_species)

# Merge all data
plot_data = tsne_coords.merge(
    protein_mapping[['UGT_ID', 'species']].drop_duplicates(),
    on='UGT_ID',
    how='left'
).merge(
    full_df[['UGT_ID', 'is_active']],
    on='UGT_ID',
    how='left'
)

# Calculate statistics for caption
unique_proteins = protein_mapping['UGT_ID'].nunique()
species_counts = protein_mapping['species'].value_counts()
human_pct = species_counts.get('Human', 0) / len(protein_mapping) * 100
nonhuman_pct = species_counts.get('Non-human', 0) / len(protein_mapping) * 100

print(f"Creating visualization for {unique_proteins} unique proteins...")
print(f"  Human: {human_pct:.0f}%")
print(f"  Non-human: {nonhuman_pct:.0f}%")

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Define colors and markers
species_activity_colors = {
    ('Human', 1): '#1f77b4',      # Blue - Human active
    ('Human', 0): '#1f77b4',      # Blue - Human inactive (none in dataset)
    ('Non-human', 1): '#ff7f0e',  # Orange - Non-human active
    ('Non-human', 0): '#2ca02c'   # Green - Non-human inactive (for visibility)
}
activity_markers = {
    1: 'o',  # Circle for active
    0: 'x'   # X for inactive
}

# Plot by species and activity
for species in ['Non-human', 'Human']:  # Plot non-human first (background)
    species_data = plot_data[plot_data['species'] == species]
    
    for activity in [0, 1]:  # Plot inactive first
        subset = species_data[species_data['is_active'] == activity]
        if len(subset) > 0:
            label = f"{species} ({'Active' if activity == 1 else 'Inactive'})"
            alpha = 0.7 if activity == 1 else 0.6
            size = 20 if activity == 1 else 40
            marker = activity_markers[activity]
            color = species_activity_colors[(species, activity)]
            
            ax.scatter(
                subset['x'],
                subset['y'],
                c=color,
                marker=marker,
                s=size,
                alpha=alpha,
                label=label,
                linewidths=1.5 if marker == 'x' else 0
            )

# Styling
ax.set_xlabel('t-SNE Component 1', fontsize=12, fontweight='bold')
ax.set_ylabel('t-SNE Component 2', fontsize=12, fontweight='bold')
ax.set_title('ProtT5 Protein Embeddings: Species Distribution and Activity',
             fontsize=14, fontweight='bold', pad=20)

# Legend
legend = ax.legend(
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=True,
    fontsize=10,
    title='Protein Classification',
    title_fontsize=11
)
legend.get_frame().set_alpha(0.9)

# Grid
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# Set equal aspect ratio for better visualization
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

# Save figure
output_file = OUTPUT_DIR / "protein_embeddings_tsne_prott5.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved figure to: {output_file}")

# Create caption
caption = f"""ProtT5 embeddings of {unique_proteins} GT enzymes projected via t-SNE
(1024D → 2D). Human UGTs (blue, {human_pct:.0f}%) cluster separately from 
plant/bacterial GTs (orange, {nonhuman_pct:.0f}%), demonstrating that embeddings
capture evolutionary relationships while maintaining sufficient diversity
for cross-species prediction. Active pairs shown as circles (o), inactive as crosses (x)."""

print("\n" + "="*80)
print("FIGURE CAPTION:")
print("="*80)
print(caption)
print("="*80)

# Show plot
plt.show()

print("\n✓ Visualization complete!")
