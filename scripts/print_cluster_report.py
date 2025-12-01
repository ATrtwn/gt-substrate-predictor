#!/usr/bin/env python
"""Print a summary report of MMseqs2 clustering results."""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPORTS = ROOT / "reports"
CLUSTER_TSV = REPORTS / "GT_cluster_cluster.tsv"

def print_report():
    """Read cluster TSV and print statistics."""
    if not CLUSTER_TSV.exists():
        print(f"Cluster file not found: {CLUSTER_TSV}")
        return
    
    # Read cluster file
    df = pd.read_csv(CLUSTER_TSV, sep='\t', header=None, names=["seq_id", "rep_id"], dtype=str)
    
    # Compute statistics
    cluster_sizes = df['rep_id'].value_counts()
    num_clusters = len(cluster_sizes)
    num_sequences = len(df)
    num_singletons = (cluster_sizes == 1).sum()
    avg_cluster_size = cluster_sizes.mean()
    max_cluster_size = cluster_sizes.max()
    min_cluster_size = cluster_sizes.min()
    
    # Top 10 clusters
    top_clusters = cluster_sizes.sort_values(ascending=False).head(10)
    
    # Print report
    report = f"""
================================================================================
                    MMseqs2 Clustering Report
================================================================================

Input file:              {CLUSTER_TSV.name}
Total sequences:         {num_sequences}
Total clusters:          {num_clusters}
Singleton clusters:      {num_singletons} ({100*num_singletons/num_clusters:.1f}%)

Cluster Size Statistics:
  Average:               {avg_cluster_size:.2f}
  Maximum:               {max_cluster_size}
  Minimum:               {min_cluster_size}

Top 10 Largest Clusters:
"""
    
    for i, (rep_id, size) in enumerate(top_clusters.items(), 1):
        report += f"  {i:2d}. {rep_id:30s} : {size:4d} sequences\n"
    
    report += f"""
Redundancy Reduction:
  Original sequences:    {num_sequences}
  Representative seqs:   {num_clusters}
  Redundancy:            {100*(1 - num_clusters/num_sequences):.1f}%

Interpretation:
  - Clustering was done with --min-seq-id 0.7 (70% identity) and -c 0.7 (70% coverage)
  - Sequences in the same cluster share ≥70% identity with ≥70% coverage
  - Use GT_cluster_rep_seq.fasta for a non-redundant set of representatives
  - Use GT_cluster_all_seqs.fasta to see all sequences grouped by cluster
  - Use GT_cluster_cluster.tsv for the cluster assignments (seq_id → representative)

================================================================================
"""
    
    print(report)
    
    # Save to file
    report_file = REPORTS / "clustering_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

if __name__ == '__main__':
    print_report()
