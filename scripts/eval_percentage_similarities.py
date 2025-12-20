import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 'Agg' backend doesn't require GUI/tkinter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
path = Path(__file__).resolve().parent.parent / "reports" / "split_fasta"


# query: Test sequence ID

# target: Training sequence ID

# pident: Percent identity ← THIS IS WHAT YOU WANT

# alnlen: Alignment length

# mismatch: Number of mismatches

# gapopen: Number of gap openings

# qstart,qend: Query start/end positions

# tstart,tend: Target start/end positions

# evalue: E-value

# bits: Bit score

# qcov: Query coverage

# tcov: Target coverage

test_c1_df = pd.read_csv(path / 'test_c1_train_comp.tsv', sep='\t', header=None,
                   names=['query', 'target', 'pident', 'alnlen', 'mismatch', 
                          'gapopen', 'qstart', 'qend', 'tstart', 'tend', 
                          'evalue', 'bits', 'qcov', 'tcov'])

test_c2_df = pd.read_csv(path / 'test_c2_train_comp.tsv', sep='\t', header=None,
                   names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                          'gapopen', 'qstart', 'qend', 'tstart', 'tend',
                          'evalue', 'bits', 'qcov', 'tcov'])
val_c1_df = pd.read_csv(path / 'val_c1_train_comp.tsv', sep='\t', header=None,
                   names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                          'gapopen', 'qstart', 'qend', 'tstart', 'tend',
                          'evalue', 'bits', 'qcov', 'tcov'])
val_c2_df = pd.read_csv(path / 'val_c2_train_comp.tsv', sep='\t', header=None,
                   names=['query', 'target', 'pident', 'alnlen', 'mismatch',
                          'gapopen', 'qstart', 'qend', 'tstart', 'tend',
                          'evalue', 'bits', 'qcov', 'tcov'])

def get_max_similarities(df, min_coverage=0.8):
    """
    Get maximum percent identity for each query sequence
    Optional: filter by coverage to avoid short fragment matches
    """
    # Filter by coverage if desired
    if min_coverage > 0:
        df = df[(df['qcov'] >= min_coverage) & (df['tcov'] >= min_coverage)]
    
    # Get maximum pident for each query
    max_sim = df.loc[df.groupby('query')['pident'].idxmax()]
    
    return max_sim[['query', 'pident', 'target', 'qcov', 'tcov', 'evalue']]

def get_percentage_similarities_fast(df, threshold=90, min_coverage=0.8):
    """
    Faster version using vectorized operations
    """
    # Filter by coverage
    if min_coverage > 0:
        df = df[(df['qcov'] >= min_coverage) & (df['tcov'] >= min_coverage)]
    
    # Group by query
    grouped = df.groupby('query')
    
    # Calculate statistics per group
    total_matches = grouped.size()
    count_above = grouped['pident'].apply(lambda x: (x >= threshold).sum())
    
    # Calculate percentage
    percentage_above = (count_above / total_matches * 100).fillna(0)
    
    # No max similarity, just percentage
    result = pd.DataFrame({
        'percentage_above': percentage_above,
        'count_above': count_above,
        'total_matches': total_matches,
    }).reset_index()
    
    return result

def calculate_leakage_stats(max_sim_df, thresholds=[30, 50, 70, 80, 90, 95, 100]):
    """Calculate percentage of sequences above identity thresholds"""
    stats = {}
    total = len(max_sim_df)
    
    for thr in thresholds:
        above = len(max_sim_df[max_sim_df['percentage_above'] >= thr])
        stats[thr] = {
            'count': above,
            'percentage': (above / total) * 100,
            'total': total
        }
    
    return stats

def bootstrap_statistic(data, statistic_func, n_bootstrap=1000, ci_level=95):
    """
    Bootstrap any statistic with confidence intervals
    
    Parameters:
    - data: array-like of values (e.g., pident values)
    - statistic_func: function to compute statistic (e.g., np.mean, lambda x: np.mean(x > 80))
    - n_bootstrap: number of bootstrap samples
    - ci_level: confidence interval level
    
    Returns:
    - observed: observed statistic
    - std_error: bootstrap standard error
    - ci: confidence interval
    - bootstrap_dist: bootstrap distribution
    """
    data = np.array(data)
    n = len(data)
    
    # Calculate observed statistic
    observed = statistic_func(data)
    
    # Bootstrap
    bootstrap_values = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        # Calculate statistic on bootstrap sample
        stat = statistic_func(bootstrap_sample)
        bootstrap_values.append(stat)
    
    bootstrap_values = np.array(bootstrap_values)
    
    # Standard error = standard deviation of bootstrap distribution
    std_error = np.std(bootstrap_values, ddof=1)
    
    # Confidence interval
    lower = np.percentile(bootstrap_values, (100 - ci_level) / 2)
    upper = np.percentile(bootstrap_values, 100 - (100 - ci_level) / 2)
    
    return {
        'observed': observed,
        'std_error': std_error,
        'ci': (lower, upper),
        'bootstrap_dist': bootstrap_values
    }

# Define statistics we want to bootstrap
def prop_above_80(x):
    """Proportion of sequences with >80% identity"""
    return np.mean(x >= 80) * 100  # As percentage

def prop_above_95(x):
    """Proportion of sequences with >95% identity"""
    return np.mean(x >= 95) * 100

def mean_similarity(x):
    """Mean percent identity"""
    return np.mean(x)

# Get percentage similarities (using get_percentage_similarities_fast)
test_c1_perc = get_percentage_similarities_fast(test_c1_df, threshold=90, min_coverage=0.8)
test_c2_perc = get_percentage_similarities_fast(test_c2_df, threshold=90, min_coverage=0.8)
val_c1_perc = get_percentage_similarities_fast(val_c1_df, threshold=90, min_coverage=0.8)
val_c2_perc = get_percentage_similarities_fast(val_c2_df, threshold=90, min_coverage=0.8)

# Calculate for all datasets
test_c1_stats = calculate_leakage_stats(test_c1_perc)
test_c2_stats = calculate_leakage_stats(test_c2_perc)
val_c1_stats = calculate_leakage_stats(val_c1_perc)
val_c2_stats = calculate_leakage_stats(val_c2_perc)

# Print results
print("C1 Leakage Analysis (Percentage Similarities):")
print("=" * 50)
for thr, data in test_c1_stats.items():
    print(f">={thr}% above threshold: {data['count']}/{data['total']} ({data['percentage']:.2f}%)")

print("\nC2 Leakage Analysis (Percentage Similarities):")
print("=" * 50)
for thr, data in test_c2_stats.items():
    print(f">={thr}% above threshold: {data['count']}/{data['total']} ({data['percentage']:.2f}%)")

print("\nVal C1 Leakage Analysis (Percentage Similarities):")
print("=" * 50)
for thr, data in val_c1_stats.items():
    print(f">={thr}% above threshold: {data['count']}/{data['total']} ({data['percentage']:.2f}%)")

print("\nVal C2 Leakage Analysis (Percentage Similarities):")
print("=" * 50)
for thr, data in val_c2_stats.items():
    print(f">={thr}% above threshold: {data['count']}/{data['total']} ({data['percentage']:.2f}%)")

for stat_name, stat_func in [("Mean Percentage Above Threshold", mean_similarity),
                             (">80% Above Threshold", prop_above_80),
                             (">95% Above Threshold", prop_above_95)]:
    
    result = bootstrap_statistic(test_c1_perc['percentage_above'].values, stat_func, n_bootstrap=1000)
    
    print(f"{stat_name}:")
    print(f"  Observed: {result['observed']:.2f}")
    print(f"  Std Error: {result['std_error']:.3f}")
    print(f"  95% CI: [{result['ci'][0]:.2f}, {result['ci'][1]:.2f}]")

# Create a comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Distribution of percentage above threshold
axes[0, 0].hist(test_c1_perc['percentage_above'], bins=50, alpha=0.5, label='C1', density=True)
axes[0, 0].hist(test_c2_perc['percentage_above'], bins=50, alpha=0.5, label='C2', density=True)
axes[0, 0].hist(val_c1_perc['percentage_above'], bins=50, alpha=0.5, label='Val C1', density=True)
axes[0, 0].hist(val_c2_perc['percentage_above'], bins=50, alpha=0.5, label='Val C2', density=True)
axes[0, 0].set_xlabel('Percentage Above Threshold')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Distribution of Percentage Above Threshold')
axes[0, 0].legend()
axes[0, 0].axvline(x=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
axes[0, 0].axvline(x=95, color='g', linestyle='--', alpha=0.5, label='95% threshold')

# 2. Cumulative distribution
sorted_test_c1 = np.sort(test_c1_perc['percentage_above'])
sorted_test_c2 = np.sort(test_c2_perc['percentage_above'])
sorted_val_c1 = np.sort(val_c1_perc['percentage_above'])
sorted_val_c2 = np.sort(val_c2_perc['percentage_above'])

axes[0, 1].plot(sorted_test_c1, np.arange(len(sorted_test_c1))/len(sorted_test_c1), label='C1')
axes[0, 1].plot(sorted_test_c2, np.arange(len(sorted_test_c2))/len(sorted_test_c2), label='C2')
axes[0, 1].plot(sorted_val_c1, np.arange(len(sorted_val_c1))/len(sorted_val_c1), label='Val C1')
axes[0, 1].plot(sorted_val_c2, np.arange(len(sorted_val_c2))/len(sorted_val_c2), label='Val C2')
axes[0, 1].set_xlabel('Percentage Above Threshold')
axes[0, 1].set_ylabel('Cumulative Proportion')
axes[0, 1].set_title('Cumulative Distribution')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Leakage by threshold
thresholds = [30, 50, 70, 80, 90, 95, 100]
c1_leakage = [np.mean(test_c1_perc['percentage_above'] >= t) * 100 for t in thresholds]
c2_leakage = [np.mean(test_c2_perc['percentage_above'] >= t) * 100 for t in thresholds]
val_c1_leakage = [np.mean(val_c1_perc['percentage_above'] >= t) * 100 for t in thresholds]
val_c2_leakage = [np.mean(val_c2_perc['percentage_above'] >= t) * 100 for t in thresholds]

axes[0, 2].plot(thresholds, c1_leakage, 'o-', label='C1')
axes[0, 2].plot(thresholds, c2_leakage, 's-', label='C2')
axes[0, 2].plot(thresholds, val_c1_leakage, 'r-', label='Val C1')
axes[0, 2].plot(thresholds, val_c2_leakage, 'g-', label='Val C2')
axes[0, 2].set_xlabel('Percentage Threshold (%)')
axes[0, 2].set_ylabel('% of Test Sequences')
axes[0, 2].set_title('Leakage by Threshold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# 4. Bootstrap distribution for mean percentage above threshold
c1_mean_boot = bootstrap_statistic(test_c1_perc['percentage_above'].values, mean_similarity, n_bootstrap=1000)
c2_mean_boot = bootstrap_statistic(test_c2_perc['percentage_above'].values, mean_similarity, n_bootstrap=1000)
val_c1_mean_boot = bootstrap_statistic(val_c1_perc['percentage_above'].values, mean_similarity, n_bootstrap=1000)
val_c2_mean_boot = bootstrap_statistic(val_c2_perc['percentage_above'].values, mean_similarity, n_bootstrap=1000)

axes[1, 0].hist(c1_mean_boot['bootstrap_dist'], bins=30, alpha=0.5, label='C1', density=True)
axes[1, 0].hist(c2_mean_boot['bootstrap_dist'], bins=30, alpha=0.5, label='C2', density=True)
axes[1, 0].hist(val_c1_mean_boot['bootstrap_dist'], bins=30, alpha=0.5, label='Val C1', density=True)
axes[1, 0].hist(val_c2_mean_boot['bootstrap_dist'], bins=30, alpha=0.5, label='Val C2', density=True)
axes[1, 0].axvline(c1_mean_boot['observed'], color='blue', linestyle='--', label='C1 Observed')
axes[1, 0].axvline(c2_mean_boot['observed'], color='orange', linestyle='--', label='C2 Observed')
axes[1, 0].axvline(val_c1_mean_boot['observed'], color='green', linestyle='--', label='Val C1 Observed')
axes[1, 0].axvline(val_c2_mean_boot['observed'], color='red', linestyle='--', label='Val C2 Observed')
axes[1, 0].set_xlabel('Mean Percentage Above Threshold')
axes[1, 0].set_ylabel('Density')
axes[1, 0].set_title('Bootstrap Distribution of Mean Percentage Above Threshold')
axes[1, 0].legend()

# 5. Comparison with confidence intervals
datasets = ['C1', 'C2', 'Val C1', 'Val C2']
means = [c1_mean_boot['observed'], c2_mean_boot['observed'], val_c1_mean_boot['observed'], val_c2_mean_boot['observed']]
errors = [c1_mean_boot['std_error'], c2_mean_boot['std_error'], val_c1_mean_boot['std_error'], val_c2_mean_boot['std_error']]
ci_lower = [c1_mean_boot['ci'][0], c2_mean_boot['ci'][0], val_c1_mean_boot['ci'][0], val_c2_mean_boot['ci'][0]]
ci_upper = [c1_mean_boot['ci'][1], c2_mean_boot['ci'][1], val_c1_mean_boot['ci'][1], val_c2_mean_boot['ci'][1]]

x_pos = np.arange(len(datasets))
axes[1, 1].errorbar(x_pos, means, yerr=errors, fmt='o', capsize=5, label='Mean ± SE')
axes[1, 1].bar(x_pos, means, alpha=0.3)
for i, (low, high) in enumerate(zip(ci_lower, ci_upper)):
    axes[1, 1].plot([i, i], [low, high], 'k-', linewidth=2)
axes[1, 1].set_xticks(x_pos)
axes[1, 1].set_xticklabels(datasets)
axes[1, 1].set_ylabel('Mean Percentage Above Threshold (%)')
axes[1, 1].set_title('Mean Percentage Above Threshold with 95% CI')
axes[1, 1].grid(True, alpha=0.3)

# 6. Summary table (text)
axes[1, 2].axis('off')
summary_text = [
    "SUMMARY STATISTICS (Percentage) ",
    "=" * 30,
    f"C1 Test Set (n={len(test_c1_perc)})",
    f"  Mean percentage above threshold: {np.mean(test_c1_perc['percentage_above']):.1f}%",
    f"  Median percentage above threshold: {np.median(test_c1_perc['percentage_above']):.1f}%",
    f"  >80% above threshold: {np.mean(test_c1_perc['percentage_above'] >= 80)*100:.1f}%",
    f"  >95% above threshold: {np.mean(test_c1_perc['percentage_above'] >= 95)*100:.1f}%",
    "",
    f"C2 Test Set (n={len(test_c2_perc)})",
    f"  Mean percentage above threshold: {np.mean(test_c2_perc['percentage_above']):.1f}%",
    f"  Median percentage above threshold: {np.median(test_c2_perc['percentage_above']):.1f}%",
    f"  >80% above threshold: {np.mean(test_c2_perc['percentage_above'] >= 80)*100:.1f}%",
    f"  >95% above threshold: {np.mean(test_c2_perc['percentage_above'] >= 95)*100:.1f}%",
    "",
    f"Val C1 Set (n={len(val_c1_perc)})",
    f"  Mean percentage above threshold: {np.mean(val_c1_perc['percentage_above']):.1f}%",
    f"  Median percentage above threshold: {np.median(val_c1_perc['percentage_above']):.1f}%",
    f"  >80% above threshold: {np.mean(val_c1_perc['percentage_above'] >= 80)*100:.1f}%",
    f"  >95% above threshold: {np.mean(val_c1_perc['percentage_above'] >= 95)*100:.1f}%",
    "",
    f"Val C2 Set (n={len(val_c2_perc)})",
    f"  Mean percentage above threshold: {np.mean(val_c2_perc['percentage_above']):.1f}%",
    f"  Median percentage above threshold: {np.median(val_c2_perc['percentage_above']):.1f}%",
    f"  >80% above threshold: {np.mean(val_c2_perc['percentage_above'] >= 80)*100:.1f}%",
    f"  >95% above threshold: {np.mean(val_c2_perc['percentage_above'] >= 95)*100:.1f}%"
]
axes[1, 2].text(0.1, 0.95, '\n'.join(summary_text), 
                verticalalignment='top',
                fontfamily='monospace',
                fontsize=10)

plt.tight_layout()
plt.savefig('leakage_analysis_percentage.png', dpi=150, bbox_inches='tight')
plt.show()

def generate_report(test_c1_perc, test_c2_perc, val_c1_perc, val_c2_perc, filename='leakage_report_percentage.txt'):
    """Generate a comprehensive report"""
    
    with open(filename, 'w') as f:
        f.write("DATA LEAKAGE ANALYSIS REPORT (Percentage Similarities)\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. DATASET OVERVIEW\n")
        f.write("-" * 30 + "\n")
        f.write(f"C1 test sequences: {len(test_c1_perc)}\n")
        f.write(f"C2 test sequences: {len(test_c2_perc)}\n")
        f.write(f"Val C1 test sequences: {len(val_c1_perc)}\n")
        f.write(f"Val C2 test sequences: {len(val_c2_perc)}\n\n")
        
        f.write("2. SIMILARITY STATISTICS\n")
        f.write("-" * 30 + "\n")
        
        for name, data in [("C1", test_c1_perc), ("C2", test_c2_perc),
                           ("Val C1", val_c1_perc), ("Val C2", val_c2_perc)]:
            f.write(f"\n{name}:\n")
            f.write(f"  Mean ± SE: {np.mean(data['percentage_above']):.2f}% ± {np.std(data['percentage_above'])/np.sqrt(len(data)):.3f}%\n")
            f.write(f"  Median: {np.median(data['percentage_above']):.2f}%\n")
            f.write(f"  Range: [{data['percentage_above'].min():.2f}%, {data['percentage_above'].max():.2f}%]\n")
            
            # Bootstrapped statistics
            for stat_name, stat_func, thr in [("Mean", mean_similarity, None),
                                             (">80%", prop_above_80, 80),
                                             (">95%", prop_above_95, 95)]:
                
                result = bootstrap_statistic(data['percentage_above'].values, stat_func, n_bootstrap=1000)
                f.write(f"  {stat_name}: {result['observed']:.2f}% ± {result['std_error']:.3f}%\n")
                f.write(f"    95% CI: [{result['ci'][0]:.2f}%, {result['ci'][1]:.2f}%]\n")
        
        f.write("\n3. LEAKAGE ASSESSMENT\n")
        f.write("-" * 30 + "\n")
        
        # Decision logic
        for name, data in [("C1", test_c1_perc), ("C2", test_c2_perc),
                           ("Val C1", val_c1_perc), ("Val C2", val_c2_perc)]:
            prop_95 = np.mean(data['percentage_above'] >= 95) * 100
            
            if prop_95 > 5:
                assessment = "SEVERE LEAKAGE - Must re-split data"
            elif prop_95 > 1:
                assessment = "MODERATE LEAKAGE - Consider re-splitting"
            elif prop_95 > 0.1:
                assessment = "MINOR LEAKAGE - Probably acceptable"
            else:
                assessment = "NO SIGNIFICANT LEAKAGE - Good split"
            
            f.write(f"\n{name}: {assessment}\n")
            f.write(f"  Rationale: {prop_95:.2f}% of test sequences are >95% above threshold\n")
        
        f.write("\n4. RECOMMENDATIONS\n")
        f.write("-" * 30 + "\n")
        f.write("1. For sequences with >95% above threshold: Remove from test set\n")
        f.write("2. For sequences with 80-95% above threshold: Consider if this level of similarity\n")
        f.write("   is problematic for your specific biological question\n")
        f.write("3. Consider using MMseqs2 clustering to create non-redundant splits\n")
        f.write("4. Document the leakage rates in your methods section\n")

# Generate the report
generate_report(test_c1_perc, test_c2_perc, val_c1_perc, val_c2_perc)
print("Report saved as 'leakage_report_percentage.txt'")
