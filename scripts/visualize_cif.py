# visualize_cif.py
import py3Dmol
import os
import sys
from pathlib import Path
def visualize_cif(cif_path: str, width: int = 800, height: int = 600):
    """
    Visualize a CIF file using py3Dmol.
    
    Args:
        cif_path (str): Path to the CIF file
        width (int): Width of the viewer
        height (int): Height of the viewer
    """
    # Read the CIF file
    with open(cif_path, 'r') as f:
        cif_data = f.read()

    # Initialize the viewer
    viewer = py3Dmol.view(width=width, height=height)
    viewer.addModel(cif_data, 'cif')  # specify format
    viewer.setStyle({'stick': {}})    # display as sticks
    viewer.zoomTo()
    viewer.write_html('visualization.html')
    os.startfile('visualization.html')

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_cif.py path/to/file.cif")
        sys.exit(1)
    cif_file = sys.argv[1]
    script_dir = Path(__file__).parent.parent
    full_path = script_dir / cif_file
    visualize_cif(full_path)
