#!/usr/bin/env python
"""Analyze ChemBERTa substrate embeddings: clusters, neighbors, outliers, plots."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False

ROOT = Path(__file__).parent.parent
EMBEDDINGS_DIR = ROOT / "embeddings"
REPORTS_DIR = ROOT / "reports"

EMB_NPY = EMBEDDINGS_DIR / "substrate_embeddings_chemberta2.npy"
EMB_CSV = EMBEDDINGS_DIR / "Substrate_with_embeddings_chemberta2.csv"
COORDS_TSNE = EMBEDDINGS_DIR / "substrate_2d_coords_tsne_chemberta2.csv"
COORDS_UMAP = EMBEDDINGS_DIR / "substrate_2d_coords_umap_chemberta2.csv"

REPORTS_DIR.mkdir(exist_ok=True, parents=True)


def load_data():
    if not EMB_NPY.exists():
        raise SystemExit(f"Embeddings not found: {EMB_NPY}. Run substrate_emb_ChamBERTA2.py first.")
    X = np.load(EMB_NPY)
    df = pd.read_csv(EMB_CSV)
    df_valid = df[df['embedding'].notna()].reset_index(drop=True)
    return df_valid, X


def get_or_make_2d_coords(X):
    if COORDS_TSNE.exists():
        coords = pd.read_csv(COORDS_TSNE)
        return coords[['x', 'y']].to_numpy(), 'tsne'
    if COORDS_UMAP.exists():
        coords = pd.read_csv(COORDS_UMAP)
        return coords[['x', 'y']].to_numpy(), 'umap'
    if not HAS_TSNE:
        raise SystemExit("No 2D coords found and t-SNE not available. Run visualize_substrate_embeddings.py first.")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, metric='cosine')
    xy = tsne.fit_transform(X)
    pd.DataFrame({'x': xy[:,0], 'y': xy[:,1]}).to_csv(COORDS_TSNE, index=False)
    return xy, 'tsne'


def choose_kmeans_k(X, k_min=2, k_max=10):
    best_k, best_score = None, -1
    for k in range(k_min, min(k_max, len(X)) + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        # avoid degenerate case
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels, metric='cosine')
        if score > best_score:
            best_k, best_score = k, score
    return best_k, best_score


def run_kmeans(X, k=None):
    if k is None:
        best_k, best_score = choose_kmeans_k(X)
        if best_k is None:
            # fallback to k=5
            best_k = 5
    else:
        best_k = k
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    # compute silhouette score for chosen k
    if len(set(labels)) >= 2:
        best_score = silhouette_score(X, labels, metric='cosine')
    else:
        best_score = -1
    return labels, best_k, best_score


def compute_neighbors(X, k=5):
    nn = NearestNeighbors(n_neighbors=min(k+1, len(X)), metric='cosine')
    nn.fit(X)
    dists, idxs = nn.kneighbors(X)
    # remove self neighbor at position 0
    return dists[:, 1:], idxs[:, 1:]


def compute_outliers(X):
    iso = IsolationForest(random_state=42, contamination='auto')
    scores = -iso.fit(X).decision_function(X)  # higher = more outlier-ish
    return scores


def main(k=None):
    sns.set_theme(context='paper', style='whitegrid', palette='pastel', font_scale=1.1)
    df, X = load_data()
    print(f"Embeddings: {X.shape}, valid substrates: {len(df)}")

    # KMeans clustering
    labels, k, sil = run_kmeans(X, k=k)
    df['cluster_kmeans'] = labels
    df.to_csv(EMBEDDINGS_DIR / 'Substrate_with_clusters_chemberta2.csv', index=False)
    counts = df['cluster_kmeans'].value_counts().sort_index()
    counts.to_csv(REPORTS_DIR / 'substrate_cluster_sizes_chemberta2.tsv', sep='\t', header=['size'])
    print(f"KMeans clusters: k={k}, silhouette={sil:.3f} (cosine)")

    # Nearest neighbors
    dists, idxs = compute_neighbors(X, k=5)
    rows = []
    names = df['substrate'].fillna('NA').astype(str).tolist()
    smiles = df.get('smiles', pd.Series(['']*len(df))).astype(str).tolist()
    for i in range(len(df)):
        for j in range(idxs.shape[1]):
            rows.append({
                'query_idx': i,
                'query': names[i],
                'query_smiles': smiles[i],
                'neighbor_rank': j+1,
                'neighbor_idx': int(idxs[i, j]),
                'neighbor': names[idxs[i, j]],
                'neighbor_smiles': smiles[idxs[i, j]],
                'cosine_distance': float(dists[i, j])
            })
    nn_df = pd.DataFrame(rows)
    nn_df.to_csv(EMBEDDINGS_DIR / 'substrate_neighbors_chemberta2.csv', index=False)
    print("Saved nearest neighbors table.")

    # Outliers
    out_scores = compute_outliers(X)
    out_df = df[['substrate', 'smiles']].copy()
    out_df['outlier_score'] = out_scores
    out_df.sort_values('outlier_score', ascending=False).head(25).to_csv(
        REPORTS_DIR / 'substrate_top_outliers_chemberta2.csv', index=False)
    print("Saved top outliers.")

    # 2D plot colored by cluster
    xy, method = get_or_make_2d_coords(X)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(xy[:,0], xy[:,1], c=labels, cmap='tab20', s=45, alpha=0.85)
    plt.title(f'ChemBERTa-2 Substrate Clusters (k={k}, {method.upper()})')
    plt.xlabel(f'{method.upper()}-1')
    plt.ylabel(f'{method.upper()}-2')
    plt.colorbar(scatter, label='k-means cluster')
    plt.tight_layout()
    out_plot = REPORTS_DIR / f'substrate_embeddings_{method}_chemberta2_clusters.png'
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"Saved cluster-colored plot to {out_plot}")

    # Small summary report
    with open(REPORTS_DIR / 'substrate_embedding_analysis_chemberta2.txt', 'w', encoding='utf-8') as f:
        f.write("ChemBERTa-2 Embedding Analysis\n")
        f.write("==============================\n\n")
        f.write(f"Embeddings shape: {X.shape}\n")
        f.write(f"KMeans k: {k}, silhouette (cosine): {sil:.3f}\n")
        f.write("\nCluster sizes:\n")
        f.write(counts.to_string())
        f.write("\n\nTop 10 outliers (higher score = more outlier):\n")
        f.write(out_df.sort_values('outlier_score', ascending=False).head(10).to_string(index=False))
    print("Saved analysis summary.")


if __name__ == '__main__':
    import sys
    k = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(k=k)
