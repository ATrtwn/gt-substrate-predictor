import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

# Constants
N_CONFIGS = 1780
N_MODELS = 5
BINDING_SITE_CUTOFF = 6.0  # Angstroms
K_NEIGHBORS = 10           # Connect to k-nearest atom neighbors

def get_atom_feature(atom):
    """
    Returns an integer representing the atom element.
    """
    element_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'CL': 6, 'BR': 7, 'I': 8}
    # Get element (Bio.PDB atom.element is usually uppercase)
    elem = atom.element.upper() if atom.element else atom.name[0].upper()
    return element_map.get(elem, 9) # 9 for 'Other'

def analyze_to_pocket_graph(cif_file_path: Path, is_active: int):
    """
    Identifies the binding pocket and converts it into an atom-level graph.
    """
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure("protein", str(cif_file_path))
    except Exception:
        return None

    ligand_atoms = []
    protein_atoms = []

    # 1. Separate Substrate (Ligands) and Enzyme (Protein)
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    protein_atoms.extend(list(residue.get_atoms()))
                elif residue.get_resname() not in ["HOH", "WAT"]:
                    ligand_atoms.extend(list(residue.get_atoms()))

    if not ligand_atoms:
        return None

    # 2. Identify Binding Pocket Atoms
    # Protein atoms within BINDING_SITE_CUTOFF of ANY ligand atom
    ns = NeighborSearch(protein_atoms)
    pocket_residues = set()
    for latom in ligand_atoms:
        neighbors = ns.search(latom.coord, BINDING_SITE_CUTOFF, level="R")
        for n in neighbors:
            pocket_residues.add(n)
    
    # All atoms from the identified pocket residues
    pocket_atoms = [atom for res in pocket_residues for atom in res.get_atoms()]
    
    # Combine ligand atoms and pocket atoms to form the graph nodes
    all_nodes_atoms = ligand_atoms + pocket_atoms
    
    if not all_nodes_atoms:
        return None

    # 3. Build Node Features and Coordinates
    coords = []
    x_features = []
    
    for atom in all_nodes_atoms:
        coords.append(atom.coord)
        
        # Feature vector: [AtomType, IsLigandFlag]
        is_ligand = 1 if atom in ligand_atoms else 0
        x_features.append([get_atom_feature(atom), is_ligand])

    coords = np.array(coords)
    x = torch.tensor(x_features, dtype=torch.float)
    y = torch.tensor([is_active], dtype=torch.long)

    # 4. Construct Edges (k-Nearest Neighbors)
    # Using sklearn for a robust k-NN implementation
    k = min(K_NEIGHBORS, len(all_nodes_atoms) - 1)
    if k < 1: return None
    
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    
    # Convert k-NN indices to PyG edge_index format [2, E]
    # We skip the first neighbor because it's the atom itself
    edge_sources = np.repeat(np.arange(len(all_nodes_atoms)), k)
    edge_targets = indices[:, 1:].flatten()
    
    edge_index = torch.tensor(np.array([edge_sources, edge_targets]), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, pos=torch.tensor(coords, dtype=torch.float), y=y)

def collect(root_dir: Path):
    # Setup paths and load labels
    data_dir = Path(__file__).parent.parent.parent / "data"
    df = pd.read_csv(data_dir / "split.csv")
    
    output_base = root_dir / "pocket_graphs"
    output_base.mkdir(parents=True, exist_ok=True)

    split_collections = {"train": [], "val": [], "test": []}

    print("Building Atom-Level Pocket Graphs...")

    for c in tqdm(range(1, N_CONFIGS + 1)):
        config_dir = root_dir / f"boltz_results_config{c}" / "predictions"
        if not config_dir.exists(): continue
        
        # Match UGT_ID to labels
        row = df.loc[df["UGT_ID"] == c]
        if row.empty: continue
            
        is_active = int(row["is_active"].values[0])
        split_name = row["dataset_split"].values[0]
        category = "val" if "val" in split_name.lower() else ("test" if "test" in split_name.lower() else "train")

        for m in range(N_MODELS):
            cif_path = config_dir / f"config{c}" / f"plddt_config{c}_model_{m}.cif"
            if cif_path.exists():
                data = analyze_to_pocket_graph(cif_path, is_active)
                if data:
                    data.config_id = c
                    data.model_id = m
                    split_collections[category].append(data)

    # Save splits
    for category, data_list in split_collections.items():
        if data_list:
            torch.save(data_list, output_base / f"{category}_pocket_dataset.pt")
            print(f"Saved {len(data_list)} pocket graphs to {category}_pocket_dataset.pt")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/boltz_results_root")
        sys.exit(1)
    collect(Path(sys.argv[1]).resolve())