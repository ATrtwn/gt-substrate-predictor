#!/usr/bin/env python
"""Visualize ChemBERTa substrate embeddings using UMAP/t-SNE."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

ROOT = Path(__file__).parent.parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"
REPORTS_DIR = ROOT / "reports"

def visualize_embeddings(method='tsne'):
    """Load ChemBERTa embeddings and create 2D visualization."""
    
    # Load embeddings
    emb_path = EMBEDDINGS_DIR / "substrate_embeddings_chemberta2.npy"
    csv_path = EMBEDDINGS_DIR / "Substrate_with_embeddings_chemberta2.csv"
    
    if not emb_path.exists():
        print(f"Embeddings not found at {emb_path}")
        print("Run: python src/features/substrate_emb_ChamBERTA2.py first")
        return
    
    embeddings = np.load(emb_path)
    df = pd.read_csv(csv_path)
    
    # Filter to only rows with valid embeddings
    df_valid = df[df['embedding'].notna()].reset_index(drop=True)
    
    print(f"Loaded {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}")
    print(f"Valid substrates: {len(df_valid)}")
    
    # Reduce to 2D
    if method == 'umap' and HAS_UMAP:
        print("Running UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)
    else:
        print("Running t-SNE...")
        tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
        embedding_2d = tsne.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                alpha=0.6, s=50, c=range(len(embedding_2d)), cmap='viridis')
    
    # Annotate some substrates
    for i in range(min(20, len(df_valid))):
        plt.annotate(df_valid.iloc[i]['substrate'], 
                    (embedding_2d[i, 0], embedding_2d[i, 1]),
                    fontsize=8, alpha=0.7)
    
    plt.title(f'ChemBERTa-2 Substrate Embeddings ({method.upper()})')
    plt.xlabel(f'{method.upper()} Dimension 1')
    plt.ylabel(f'{method.upper()} Dimension 2')
    plt.colorbar(label='Substrate Index')
    plt.tight_layout()
    
    output_path = REPORTS_DIR / f"substrate_embeddings_{method}_chemberta2.png"
    plt.savefig(output_path, dpi=300)
    print(f"Saved visualization to {output_path}")
    plt.close()
    
    # Save 2D coordinates
    df_valid['x'] = embedding_2d[:, 0]
    df_valid['y'] = embedding_2d[:, 1]
    coord_path = EMBEDDINGS_DIR / f"substrate_2d_coords_{method}_chemberta2.csv"
    df_valid[['substrate', 'smiles', 'x', 'y']].to_csv(coord_path, index=False)
    print(f"Saved 2D coordinates to {coord_path}")

if __name__ == '__main__':
    import sys
    method = sys.argv[1] if len(sys.argv) > 1 else 'tsne'
    visualize_embeddings(method=method)
