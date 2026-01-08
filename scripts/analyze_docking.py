# visualize_cif.py
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
import py3Dmol
import os



def analyze(cif_file: str, folder: str, model: int):
    """
    Visualize a CIF file using py3Dmol.
    """
    
    script_dir = Path(__file__).parent.parent
    file_path = script_dir / folder /cif_file
    folder_path = script_dir / folder
    parser = MMCIFParser(QUIET=True)

    structure = parser.get_structure("protein", file_path)

    # Separate ligands from protein residues
    ligands = []
    protein_atoms = []
    protein_residues = []

    pae_path = Path(folder_path) / f"pae_{folder}_model_{model}.npz"
    pde_path = Path(folder_path) / f"pde_{folder}_model_{model}.npz"

    with np.load(pae_path, mmap_mode="r") as pae_npz:
        pae = pae_npz["pae"]
    with np.load(pde_path, mmap_mode="r") as pde_npz:
        pde = pde_npz["pde"]



    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    protein_atoms.extend(list(residue.get_atoms()))
                    protein_residues.append(residue)
                else:
                    # Exclude water
                    if residue.get_resname() not in ["HOH", "WAT"]:
                        ligands.append(residue)

    print(f"Found {len(ligands)} ligand(s):")
    for ligand in ligands:
        print(f"{ligand.get_resname()} in chain {ligand.get_parent().id}, residue number {ligand.id[1]}")

    # Find binding site: residues within a cutoff distance to any ligand atom
    binding_site_cutoff = 6.0  # Angstroms
    ns = NeighborSearch(protein_atoms)
    binding_site_residues = set()

    for ligand in ligands:
        ligand_atoms = list(ligand.get_atoms())
        close_residues = set()
        for atom in ligand_atoms:
            neighbors = ns.search(atom.coord, binding_site_cutoff, level="R")
            for neighbor in neighbors:
                close_residues.add((neighbor.get_resname(), neighbor.get_parent().id, neighbor.id[1]))
                binding_site_residues.add(neighbor)

        print(f"\nBinding site residues for ligand {ligand.get_resname()}:")
        for res in sorted(close_residues):
            print(f"Residue {res[0]} chain {res[1]} number {res[2]}")

    binding_site_indices = [i for i, res in enumerate(protein_residues) if res in binding_site_residues]
    print(f"\nBinding site residue indices: {binding_site_indices}")
    pae_binding_site = pae[np.ix_(binding_site_indices, binding_site_indices)]
    pde_binding_site = pde[np.ix_(binding_site_indices, binding_site_indices)]

    # --- Prepare Py3Dmol visualization ---
    view = py3Dmol.view(width=800, height=600)
    # Load structure
    with open(file_path, "r") as f:
        pdb_text = f.read()
    view.addModel(pdb_text, "cif")

    # Style: protein default cartoon
    view.setStyle({'chain':'A'}, {'cartoon':{'color':'lightgrey'}})

    # Highlight binding site residues
    for _, chain_id, resnum in close_residues:
        view.setStyle({'chain': chain_id, 'resi': resnum}, {'stick': {'color':'red'}})

    # Highlight ligand residues
    for ligand in ligands:
        resnum = ligand.id[1]
        chain_id = ligand.get_parent().id
        view.setStyle({'chain': chain_id, 'resi': resnum}, {'stick': {'color':'green'}})

    view.zoomTo()

    view.write_html('visualization.html')
    os.startfile('visualization.html')

    plt.imshow(pae_binding_site, cmap="viridis")
    plt.colorbar(label="PAE (Å)")
    plt.title("Predicted Aligned Error")
    plt.show()

    plt.imshow(pde_binding_site, cmap="viridis")
    plt.colorbar(label="PDE (Å)")
    plt.title("Predicted Distance Error")
    plt.show()

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python visualize_cif.py path/to/file.cif")
        sys.exit(1)
    folder = sys.argv[1]
    cif_file = sys.argv[2]
    model = sys.argv[3]
    analyze(cif_file=cif_file,  folder=folder, model=int(model))
    
