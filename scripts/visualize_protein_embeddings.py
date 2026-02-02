#!/usr/bin/env python
"""Visualize ProtT5 protein embeddings using t-SNE and UMAP dimensionality reduction."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = ROOT / "data" / "Protein_Embeddings"
REPORTS_DIR = ROOT / "reports"

EMB_NPY = EMBEDDINGS_DIR / "protein_embeddings_prott5.npy"
MAPPING_CSV = EMBEDDINGS_DIR / "protein_embedding_mapping.csv"
REPORTS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    """Load protein embeddings and mapping information."""
    if not EMB_NPY.exists():
        raise SystemExit(f"Protein embeddings not found: {EMB_NPY}. Run protein_emb.py first.")
    if not MAPPING_CSV.exists():
        raise SystemExit(f"Mapping file not found: {MAPPING_CSV}")
    
    X = np.load(EMB_NPY)
    df = pd.read_csv(MAPPING_CSV)
    
    print(f"Loaded {X.shape[0]} protein embeddings of dimension {X.shape[1]}")
    print(f"Mapping info: {len(df)} rows")
    
    return X, df


def compute_tsne(X, perplexity=30, random_state=42):
    """Compute t-SNE dimensionality reduction."""
    print(f"Computing t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                metric='cosine', verbose=1)
    xy = tsne.fit_transform(X)
    print("t-SNE complete.")
    return xy


def compute_umap_reduction(X, n_neighbors=15, min_dist=0.1, random_state=42):
    """Compute UMAP dimensionality reduction."""
    if not HAS_UMAP:
        print("UMAP not available. Install with: pip install umap-learn")
        return None
    
    print(f"Computing UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist,
                       metric='cosine', random_state=random_state, verbose=True)
    xy = reducer.fit_transform(X)
    print("UMAP complete.")
    return xy


def plot_embeddings(xy, df, method='t-SNE', color_by='UGT_Nomenclature'):
    """Create visualization of 2D protein embeddings."""
    # Extract unique protein families or groups for coloring
    if color_by in df.columns:
        # For missing nomenclature, use UGT_ID instead
        df['label'] = df[color_by].copy()
        if 'UGT_ID' in df.columns:
            # Fill missing nomenclature with UGT_ID
            missing_mask = df['label'].isna()
            df.loc[missing_mask, 'label'] = 'ID_' + df.loc[missing_mask, 'UGT_ID'].astype(str)
        else:
            df['label'] = df['label'].fillna('Unknown')
        
        # Extract protein family prefix (e.g., UGT1A10 -> UGT1A, or ID_1349 -> ID_1349)
        df['protein_family'] = df['label'].astype(str).str.extract(r'^(ID_\d+|[A-Z]+\d*[A-Z]*)')[0]
        
        # Count unique families
        unique_families = df['protein_family'].nunique()
        print(f"Found {unique_families} unique protein families (including {df['protein_family'].str.startswith('ID_').sum()} by UGT_ID)")
        
        # If too many families, color by major groups only
        if unique_families > 20:
            family_counts = df['protein_family'].value_counts()
            top_families = family_counts.head(15).index.tolist()
            df['plot_group'] = df['protein_family'].apply(lambda x: x if x in top_families else 'Other')
            color_col = 'plot_group'
        else:
            color_col = 'protein_family'
    else:
        # Default: no coloring
        df['plot_group'] = 'Protein'
        color_col = 'plot_group'
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Get unique groups and assign colors
    unique_groups = df[color_col].unique()
    if len(unique_groups) <= 20:
        palette = sns.color_palette('tab20', n_colors=len(unique_groups))
    else:
        palette = sns.color_palette('husl', n_colors=len(unique_groups))
    
    # Plot each group
    for i, group in enumerate(unique_groups):
        mask = df[color_col] == group
        plt.scatter(xy[mask, 0], xy[mask, 1], 
                   c=[palette[i]], label=group, 
                   s=30, alpha=0.7, edgecolors='none')
    
    plt.title(f'ProtT5 Protein Embeddings ({method})', fontsize=14, fontweight='bold')
    plt.xlabel(f'{method}-1', fontsize=12)
    plt.ylabel(f'{method}-2', fontsize=12)
    
    # Add legend if not too many groups
    if len(unique_groups) <= 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                  fontsize=9, frameon=True, title='Protein Family')
    
    plt.tight_layout()
    
    # Save figure
    output_path = REPORTS_DIR / f"protein_embeddings_{method.lower()}_prott5.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    
    return output_path


def save_coords(xy, df, method='tsne'):
    """Save 2D coordinates to CSV."""
    coords_df = pd.DataFrame({
        'x': xy[:, 0],
        'y': xy[:, 1]
    })
    
    # Add protein identifiers if available
    if 'UGT_ID' in df.columns:
        coords_df['UGT_ID'] = df['UGT_ID'].values
    if 'UGT_Nomenclature' in df.columns:
        coords_df['UGT_Nomenclature'] = df['UGT_Nomenclature'].values
    
    output_path = EMBEDDINGS_DIR / f"protein_2d_coords_{method}_prott5.csv"
    coords_df.to_csv(output_path, index=False)
    print(f"Saved coordinates to {output_path}")
    
    return output_path


def generate_summary_report(X, df, tsne_coords=None, umap_coords=None):
    """Generate a text summary of the protein embedding analysis."""
    report_lines = [
        "ProtT5 Protein Embedding Analysis",
        "=" * 50,
        "",
        f"Embeddings shape: {X.shape}",
        f"Total proteins: {len(df)}",
        ""
    ]
    
    # Check for duplicates in protein sequences
    if 'prot_seq' in df.columns:
        unique_seqs = df['prot_seq'].nunique()
        report_lines.append(f"Unique sequences: {unique_seqs}")
        if unique_seqs < len(df):
            report_lines.append(f"Duplicate sequences: {len(df) - unique_seqs}")
    
    report_lines.append("")
    
    # Add family statistics if available
    if 'UGT_Nomenclature' in df.columns:
        df_temp = df.copy()
        # For missing nomenclature, use UGT_ID instead
        df_temp['label'] = df_temp['UGT_Nomenclature'].copy()
        if 'UGT_ID' in df_temp.columns:
            missing_mask = df_temp['label'].isna()
            df_temp.loc[missing_mask, 'label'] = 'ID_' + df_temp.loc[missing_mask, 'UGT_ID'].astype(str)
        else:
            df_temp['label'] = df_temp['label'].fillna('Unknown')
        
        # Extract family
        df_temp['family'] = df_temp['label'].astype(str).str.extract(r'^(ID_\d+|[A-Z]+\d*[A-Z]*)')[0]
        family_counts = df_temp['family'].value_counts()
        
        # Count named vs ID-based (handle potential NaN in family column)
        df_temp['family'] = df_temp['family'].fillna('Unknown')
        named_count = (~df_temp['family'].str.startswith('ID_')).sum()
        id_based_count = (df_temp['family'].str.startswith('ID_')).sum()
        
        report_lines.append(f"Named proteins: {named_count}")
        report_lines.append(f"Proteins labeled by ID (no nomenclature): {id_based_count}")
        report_lines.append("")
        
        report_lines.append("Top 10 Protein Families:")
        for family, count in family_counts.head(10).items():
            report_lines.append(f"  {family}: {count}")
        report_lines.append("")
    
    # Add dimensionality reduction info
    if tsne_coords is not None:
        report_lines.append("t-SNE: computed (2D)")
    if umap_coords is not None:
        report_lines.append("UMAP: computed (2D)")
    
    report_text = "\n".join(report_lines)
    
    # Save to file
    report_path = REPORTS_DIR / "protein_embedding_analysis_prott5.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"\nSaved analysis report to {report_path}")
    print(report_text)
    
    return report_path


def main():
    """Main function to run protein embedding visualization."""
    sns.set_theme(context='paper', style='whitegrid', font_scale=1.0)
    
    # Load data
    X, df = load_data()
    
    # Compute t-SNE
    tsne_coords = compute_tsne(X, perplexity=30)
    save_coords(tsne_coords, df, method='tsne')
    plot_embeddings(tsne_coords, df, method='t-SNE')
    
    # Compute UMAP if available
    umap_coords = None
    if HAS_UMAP:
        umap_coords = compute_umap_reduction(X, n_neighbors=15)
        if umap_coords is not None:
            save_coords(umap_coords, df, method='umap')
            plot_embeddings(umap_coords, df, method='UMAP')
    else:
        print("\nSkipping UMAP (not installed). Install with: pip install umap-learn")
    
    # Generate summary report
    generate_summary_report(X, df, tsne_coords=tsne_coords, umap_coords=umap_coords)
    
    print("\n✓ Protein embedding visualization complete!")


if __name__ == "__main__":
    main()
