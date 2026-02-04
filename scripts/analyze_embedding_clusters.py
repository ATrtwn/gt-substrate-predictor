#!/usr/bin/env python
"""Analyze spatial clustering of protein families in t-SNE/UMAP space."""
import pandas as pd
import numpy as np

# Load data
coords_tsne = pd.read_csv('data/Protein_Embeddings/protein_2d_coords_tsne_prott5.csv')
coords_umap = pd.read_csv('data/Protein_Embeddings/protein_2d_coords_umap_prott5.csv')
df = pd.read_csv('data/Protein_Embeddings/protein_embedding_mapping.csv')

# Create family labels
df['label'] = df['UGT_Nomenclature'].fillna('ID_' + df['UGT_ID'].astype(str))
df['family'] = df['label'].str.extract(r'^(ID_\d+|[A-Z]+\d*[A-Z]*)')[0].fillna('Unknown')

print("="*70)
print("PROTEIN FAMILY CLUSTERING ANALYSIS")
print("="*70)

# Analyze t-SNE clustering
print("\n📊 t-SNE SPATIAL DISTRIBUTION:")
print("-" * 70)

human_ugts = ['UGT1A', 'UGT2B', 'UGT3A', 'UGT2A']
plant_gts = ['BILIQTL1', 'UGT84A', 'UGT88A', 'UGT71A', 'UGT73E', 'UGT85A']

print("\n🧬 HUMAN UGT FAMILIES:")
for fam in human_ugts:
    mask = df['family'] == fam
    count = mask.sum()
    if count > 0:
        subset = coords_tsne.iloc[mask.values]
        x_range = subset['x'].max() - subset['x'].min()
        y_range = subset['y'].max() - subset['y'].min()
        x_center = subset['x'].mean()
        y_center = subset['y'].mean()
        print(f"  {fam:8s}: {count:4d} proteins | Center: ({x_center:6.1f}, {y_center:6.1f}) | Spread: {x_range:5.1f} × {y_range:5.1f}")

print("\n🌿 PLANT GT FAMILIES:")
for fam in plant_gts:
    mask = df['family'] == fam
    count = mask.sum()
    if count > 0:
        subset = coords_tsne.iloc[mask.values]
        x_range = subset['x'].max() - subset['x'].min()
        y_range = subset['y'].max() - subset['y'].min()
        x_center = subset['x'].mean()
        y_center = subset['y'].mean()
        print(f"  {fam:8s}: {count:4d} proteins | Center: ({x_center:6.1f}, {y_center:6.1f}) | Spread: {x_range:5.1f} × {y_range:5.1f}")

# Calculate separation between human and plant clusters
human_mask = df['family'].isin(human_ugts)
plant_mask = df['family'].isin(plant_gts)

if human_mask.sum() > 0 and plant_mask.sum() > 0:
    human_center = coords_tsne.iloc[human_mask.values][['x', 'y']].mean()
    plant_center = coords_tsne.iloc[plant_mask.values][['x', 'y']].mean()
    distance = np.sqrt((human_center['x'] - plant_center['x'])**2 + 
                      (human_center['y'] - plant_center['y'])**2)
    
    print(f"\n📏 SEPARATION:")
    print(f"  Human GT center:  ({human_center['x']:6.1f}, {human_center['y']:6.1f})")
    print(f"  Plant GT center:  ({plant_center['x']:6.1f}, {plant_center['y']:6.1f})")
    print(f"  Distance:         {distance:6.1f} units")

# Check if families overlap or are distinct
print("\n🎯 CLUSTERING QUALITY:")
print("  (Lower spread = tighter clustering)")

all_families = human_ugts + plant_gts
spreads = []
for fam in all_families:
    mask = df['family'] == fam
    if mask.sum() > 10:  # Only for families with enough members
        subset = coords_tsne.iloc[mask.values]
        spread = np.sqrt(subset[['x', 'y']].var().mean())
        spreads.append((fam, spread))

spreads.sort(key=lambda x: x[1])
print("\n  Tightest clusters (most similar sequences):")
for fam, spread in spreads[:5]:
    print(f"    {fam:12s}: spread = {spread:5.1f}")

print("\n  Loosest clusters (more diverse sequences):")
for fam, spread in spreads[-5:]:
    print(f"    {fam:12s}: spread = {spread:5.1f}")

# Unnamed proteins analysis
id_mask = df['family'].str.startswith('ID_')
if id_mask.sum() > 0:
    id_coords = coords_tsne.iloc[id_mask.values]
    print(f"\n🔍 UNNAMED PROTEINS (ID_XXXX):")
    print(f"  Count: {id_mask.sum()}")
    print(f"  Spread: x={id_coords['x'].max() - id_coords['x'].min():.1f}, y={id_coords['y'].max() - id_coords['y'].min():.1f}")
    print(f"  → Scattered across the map (diverse uncharacterized proteins)")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("✓ Check the t-SNE/UMAP visualizations to see if:")
print("  1. Same-colored points (same family) cluster together")
print("  2. Human UGTs form distinct regions from plant GTs")
print("  3. ID_XXXX proteins are scattered (diverse origins)")
print("  4. Similar families (e.g., UGT1A, UGT2B) are nearby")
print("="*70)
