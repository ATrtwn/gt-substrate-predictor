# visualize_cif.py
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

def analyze(path: str,folder: str, model: int):
    """
    Visualize a CIF file using py3Dmol.
    """
    pae_path = Path(path) / f"pae_{folder}_model_{model}.npz"
    pde_path = Path(path) / f"pde_{folder}_model_{model}.npz"
    with np.load(pae_path, mmap_mode="r") as pae_npz:
        pae = pae_npz["pae"]
    with np.load(pde_path, mmap_mode="r") as pde_npz:
        pde = pde_npz["pde"]

    import matplotlib.pyplot as plt

    plt.imshow(pae, cmap="viridis")
    plt.colorbar(label="PAE (Å)")
    plt.title("Predicted Aligned Error")
    plt.show()

    plt.imshow(pde, cmap="viridis")
    plt.colorbar(label="PDE (Å)")
    plt.title("Predicted Distance Error")
    plt.show()


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_cif.py path/to/file.cif")
        sys.exit(1)
    script_dir = Path(__file__).parent.parent
    path = sys.argv[1]
    folder = Path(path).name
    model = sys.argv[2]
    analyze(script_dir / path, folder, int(model))
    
