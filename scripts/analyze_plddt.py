import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm import tqdm

N_CONFIGS = 1
N_MODELS = 1
PLDDT_KEY = "plddt"


def analyze(root_dir: Path):
    """
    Ensemble analysis of Boltz pLDDT outputs:
    - Mean pLDDT per residue across all configs and models
    - Best-model pLDDT per residue (highest global mean over full trio)
    - Confidence variability per residue
    - Saves publication-ready figures + raw data to disk
    """

    all_profiles = []
    global_scores = []

    best_profile = None
    best_score = -np.inf
    best_ref = None

    output_dir = root_dir / "plddt_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Scanning Boltz results...")

    for c in tqdm(range(1, N_CONFIGS + 1)):
        config_dir = root_dir / f"boltz_results_config{c}" / "predictions"

        if not config_dir.exists():
            print(f"Warning: missing {config_dir}")
            continue

        for m in range(N_MODELS):
            f = config_dir / f"config{c}" / f"plddt_config{c}_model_{m}.npz"
            f = root_dir / f"plddt_config{c}_model_{m}.npz"
            if not f.exists():
                print(f"Warning: missing {f}")
                continue

            with np.load(f, mmap_mode="r") as npz:
                plddt = npz[PLDDT_KEY].astype(np.float32)

            all_profiles.append(plddt)

            global_mean = float(plddt.mean())
            global_scores.append(global_mean)

            if global_mean > best_score:
                best_score = global_mean
                best_profile = plddt.copy()
                best_ref = (c, m)

    if not all_profiles:
        print("ERROR: No pLDDT files found.")
        return

    # --- Handle variable-length profiles safely ---
    max_len = max(len(p) for p in all_profiles)
    padded_profiles = np.full((len(all_profiles), max_len), np.nan, dtype=np.float32)
    for i, p in enumerate(all_profiles):
        padded_profiles[i, :len(p)] = p

    mean_profile = np.nanmean(padded_profiles, axis=0)
    std_profile = np.nanstd(padded_profiles, axis=0)

    print("\nBest model found:")
    print(f"  Config: {best_ref[0]}")
    print(f"  Model:  {best_ref[1]}")
    print(f"  Global mean pLDDT: {best_score:.2f}")

    # --- Save raw data for reproducibility ---
    np.savez(
        output_dir / "plddt_profiles.npz",
        mean_profile=mean_profile,
        std_profile=std_profile,
        best_profile=best_profile,
        global_scores=np.array(global_scores),
        best_config=best_ref[0],
        best_model=best_ref[1],
    )

    visualize(mean_profile, best_profile, std_profile, global_scores, output_dir)


def visualize(mean_profile, best_profile, std_profile, global_scores, output_dir):
    residues = np.arange(1, len(mean_profile) + 1)

    # --- Main comparison plot ---
    plt.figure(figsize=(14, 5))
    plt.plot(residues, mean_profile, label="Ensemble Mean pLDDT", linewidth=2)
    plt.plot(residues[:len(best_profile)], best_profile, label="Best Model pLDDT", linestyle="--", linewidth=2)
    plt.fill_between(residues, mean_profile - std_profile, mean_profile + std_profile, alpha=0.2, label="±1 Std Dev")
    plt.xlabel("Residue Index")
    plt.ylabel("pLDDT")
    plt.title("Residue-wise Confidence: Ensemble vs Best Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plddt_ensemble_vs_best.png", dpi=300)
    plt.close()

    # --- Variability heatmap ---
    plt.figure(figsize=(12, 3))
    plt.imshow(std_profile[None, :], aspect="auto", cmap="viridis")
    plt.colorbar(label="pLDDT Std Dev")
    plt.yticks([])
    plt.xlabel("Residue Index")
    plt.title("Residue-wise Confidence Variability Across All Models")
    plt.tight_layout()
    plt.savefig(output_dir / "plddt_variability_heatmap.png", dpi=300)
    plt.close()

    # --- Global confidence distribution ---
    plt.figure(figsize=(6, 4))
    plt.hist(global_scores, bins=50)
    plt.axvline(max(global_scores), linestyle="--", label="Best Model")
    plt.xlabel("Global Mean pLDDT")
    plt.ylabel("Count")
    plt.title("Distribution of Global Model Confidence")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plddt_global_distribution.png", dpi=300)
    plt.close()

    print(f"\nPlots and data saved to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_plddt_ensemble.py /path/to/boltz_results_root")
        sys.exit(1)

    root_dir = Path(sys.argv[1]).resolve()
    analyze(root_dir)
