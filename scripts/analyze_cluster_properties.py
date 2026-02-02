#!/usr/bin/env python
"""Analyze relationship between ChemBERTa clusters and chemical/activity properties."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

ROOT = Path(__file__).parent.parent

DATA_DIR = ROOT / "data"
EMBEDDINGS_DIR = ROOT / "data" / "Substrate_Embeddings"
REPORTS_DIR = ROOT / "reports"

FULL_DATASET_CSV = DATA_DIR / "full_dataset.csv"
CLUSTERS_CSV = EMBEDDINGS_DIR / "Substrate_with_clusters_chemberta2.csv"
CLUSTER_SIZES_TSV = REPORTS_DIR / "substrate_cluster_sizes_chemberta2.tsv"

sns.set_theme(context='paper', style='whitegrid', palette='pastel', font_scale=1.1)


def load_data():
    """Load cluster assignments and activity data."""
    clusters = pd.read_csv(CLUSTERS_CSV)
    # Ensure cluster_kmeans matches the clusters in the cluster sizes file
    cluster_sizes = pd.read_csv(CLUSTER_SIZES_TSV, sep='\t')
    # Filter clusters to only those present in the cluster sizes file (should always match, but for safety)
    clusters = clusters[clusters['cluster_kmeans'].isin(cluster_sizes['cluster_kmeans'])]
    full_data = pd.read_csv(FULL_DATASET_CSV)
    return clusters, full_data


def merge_activity(clusters, activity):
    """Merge cluster data with activity labels."""
    # Use 'is_active' as activity, group by substrate
    # Ensure one row per substrate in act_counts and act_mode
    act_counts = activity.groupby('substrate').size().reset_index(name='num_enzymes').drop_duplicates(subset=['substrate'])
    act_mode = activity.groupby('substrate')['is_active'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'unknown').reset_index().drop_duplicates(subset=['substrate'])
    act_mode.columns = ['substrate', 'activity_mode']

    # Merge only with deduplicated summaries to prevent duplicates
    merged = clusters.merge(act_counts, on='substrate', how='left')
    merged = merged.merge(act_mode, on='substrate', how='left')
    merged['num_enzymes'] = merged['num_enzymes'].fillna(0).astype(int)
    merged['activity_mode'] = merged['activity_mode'].fillna('no_data')

    # Check for duplicates (should be zero)
    n_dupes = merged.duplicated(subset=['substrate']).sum()
    if n_dupes > 0:
        print(f"Warning: {n_dupes} duplicate substrate entries after merge.")

    return merged


def compute_rdkit_properties(df):
    """Compute chemical descriptors for substrates with SMILES."""
    properties = []
    for idx, row in df.iterrows():
        smiles = row.get('smiles', '')
        if pd.isna(smiles) or smiles == '':
            properties.append({
                'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
                'TPSA': np.nan, 'RotBonds': np.nan, 'AromaticRings': np.nan
            })
            continue
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                properties.append({
                    'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
                    'TPSA': np.nan, 'RotBonds': np.nan, 'AromaticRings': np.nan
                })
                continue
            
            properties.append({
                'MW': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Lipinski.NumRotatableBonds(mol),
                'AromaticRings': Lipinski.NumAromaticRings(mol)
            })
        except Exception:
            properties.append({
                'MW': np.nan, 'LogP': np.nan, 'HBD': np.nan, 'HBA': np.nan,
                'TPSA': np.nan, 'RotBonds': np.nan, 'AromaticRings': np.nan
            })
    
    props_df = pd.DataFrame(properties)
    return pd.concat([df, props_df], axis=1)


def plot_cluster_properties(df):
    """Plot distribution of chemical properties by cluster."""
    prop_cols = ['MW', 'LogP', 'HBD', 'HBA', 'TPSA', 'RotBonds', 'AromaticRings']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, prop in enumerate(prop_cols):
        ax = axes[i]
        df_valid = df[df[prop].notna()]
        if len(df_valid) > 0:
            sns.boxplot(data=df_valid, x='cluster_kmeans', y=prop, ax=ax, palette='Set2')
            ax.set_title(f'{prop} by Cluster')
            ax.set_xlabel('Cluster')
            ax.set_ylabel(prop)
    
    # Activity distribution
    ax = axes[7]
    activity_counts = df.groupby(['cluster_kmeans', 'activity_mode']).size().unstack(fill_value=0)
    activity_counts.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
    ax.set_title('Activity Distribution by Cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Count')
    ax.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Number of enzymes
    ax = axes[8]
    df_valid = df[df['num_enzymes'] > 0]
    if len(df_valid) > 0:
        sns.boxplot(data=df_valid, x='cluster_kmeans', y='num_enzymes', ax=ax, palette='Set2')
        ax.set_title('Enzyme Count by Cluster')
        ax.set_xlabel('Cluster')
        ax.set_ylabel('# Enzymes')
    
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'cluster_property_analysis_chemberta2.png', dpi=300)
    plt.close()
    print(f"Saved property plots to {REPORTS_DIR / 'cluster_property_analysis_chemberta2.png'}")


def analyze_cluster_enrichment(df):
    """Statistical analysis of cluster-activity relationships."""
    results = []
    
    # Chi-square test for activity distribution
    activity_ct = pd.crosstab(df['cluster_kmeans'], df['activity_mode'])
    if activity_ct.shape[0] > 1 and activity_ct.shape[1] > 1:
        chi2, p_val, dof, expected = chi2_contingency(activity_ct)
        results.append(f"Activity vs Cluster: χ²={chi2:.2f}, p={p_val:.4f}")
    
    # Summary statistics per cluster
    summary = df.groupby('cluster_kmeans').agg({
        'MW': ['mean', 'std'],
        'LogP': ['mean', 'std'],
        'HBD': ['mean', 'std'],
        'HBA': ['mean', 'std'],
        'TPSA': ['mean', 'std'],
        'num_enzymes': ['mean', 'max'],
        'substrate': 'count'
    }).round(2)
    
    return results, summary


def main():
    print("Loading data...")
    clusters, activity = load_data()
    
    print("Merging activity data...")
    df = merge_activity(clusters, activity)
    
    print("Computing RDKit properties...")
    df = compute_rdkit_properties(df)
    
    print("Analyzing cluster enrichment...")
    stats, summary = analyze_cluster_enrichment(df)
    
    print("\nStatistical Tests:")
    for s in stats:
        print(f"  {s}")
    
    print("\nCluster Summary Statistics:")
    print(summary)
    
    print("\nGenerating plots...")
    plot_cluster_properties(df)
    
    # Save enriched dataset
    df.to_csv(EMBEDDINGS_DIR / 'Substrate_with_properties_chemberta2.csv', index=False)
    print(f"Saved enriched data to {EMBEDDINGS_DIR / 'Substrate_with_properties_chemberta2.csv'}")
    
    # Save summary report
    with open(REPORTS_DIR / 'cluster_property_summary_chemberta2.txt', 'w', encoding='utf-8') as f:
        f.write("ChemBERTa-2 Cluster Property Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write("Statistical Tests:\n")
        for s in stats:
            f.write(f"  {s}\n")
        f.write("\n\nCluster Summary:\n")
        f.write(summary.to_string())
        f.write("\n\nActivity Mode Distribution:\n")
        f.write(pd.crosstab(df['cluster_kmeans'], df['activity_mode']).to_string())
    
    print(f"Saved summary to {REPORTS_DIR / 'cluster_property_summary_chemberta2.txt'}")


if __name__ == '__main__':
    main()
