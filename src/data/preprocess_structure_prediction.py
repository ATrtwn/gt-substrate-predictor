import sys
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from Bio.PDB import MMCIFParser, NeighborSearch
from Bio.PDB.Polypeptide import is_aa
from torch_geometric.data import Data
import torch
from sklearn.neighbors import NearestNeighbors
import json
proj_root = Path(__file__).resolve().parents[2]
sys.path.append(str(proj_root))
from src.features.feature_utils import compute_all_features

boltz_dir = Path(__file__).parent.parent.parent.parent/ "boltz_output" 
split_dir = Path(__file__).parent.parent / "data"/ "splits.csv"
# Constants
N_MODELS = 5
BINDING_SITE_CUTOFF = 6.0  # Angstroms
K_NEIGHBORS = 10           # Connect to k-nearest atom neighbors
REGISTRY_PATH = Path(__file__).parent.parent.parent.parent / "substrate_registry.json"

def load_registry():
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}

import json
from pathlib import Path
from typing import List, Tuple, Dict

def select_best_structure(json_files: List[Path], iptm_threshold: float = 0.75) -> Tuple[Path, Dict[str, Dict[str, float]]]:
    """
    Selects the best protein complex structure from a list of Boltz2 JSON outputs.

    Parameters:
        json_files (List[Path]): List of 5 JSON file paths.
        iptm_threshold (float): Minimum ipTM to consider an interaction significant.

    Returns:
        Tuple[Path, Dict[str, Dict[str, float]]]:
            - Path to the best JSON file
            - Dictionary of interacting chain pairs above threshold for that structure
    """
    
    def score_model(js: dict) -> float:
        """Compute a single ranking score for a JSON model."""
        pair_iptm = js["pair_chains_iptm"]
        off_diag_scores = [
            val for i, row in pair_iptm.items()
            for j, val in row.items()
            if i != j  # skip self
        ]
        mean_iptm = sum(off_diag_scores) / len(off_diag_scores)
        # Weighted combination with complex_plddt
        return 0.7 * mean_iptm + 0.3 * js.get("complex_plddt", 0)
    
    best_score = -1
    best_file = None
    best_json = None
    
    for fpath in json_files:
        if not fpath.exists():
            continue
        with open(fpath) as f:
            js = json.load(f)
        s = score_model(js)
        if s > best_score:
            best_score = s
            best_file = fpath
            best_json = js
    if best_json is None:
        print("Could not find best model")
        for fpath in json_files:
            print(f"Path : {fpath}")
            if fpath.exists():
                with open(fpath) as f:
                    js = json.load(f)
            else:
                print("Path do not exist")
                print(f"Score : {score_model(js)}")
    # Extract interacting chain pairs above threshold
    interacting_pairs = {}
    if best_json is not None:
        for i, row in best_json["pair_chains_iptm"].items():
            for j, val in row.items():
                if i != j and val >= iptm_threshold:
                    interacting_pairs.setdefault(i, {})[j] = val
    
    return best_file, interacting_pairs


def get_atom_feature(atom):
    """
    Returns an integer representing the atom element.
    """
    element_map = {'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'F': 5, 'CL': 6, 'BR': 7, 'I': 8}
    # Get element (Bio.PDB atom.element is usually uppercase)
    elem = atom.element.upper() if atom.element else atom.name[0].upper()
    return element_map.get(elem, 9) # 9 for 'Other'

def analyze_to_pocket_graph(cif_file_path: Path, is_active: int,c:str, model_str:str, UGT_ID:int, smiles:str, raw_scalars: torch.Tensor):
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

    return Data(x=x, edge_index=edge_index, pos=torch.tensor(coords, dtype=torch.float), y=y,c=c, model_str=model_str, UGT_ID=UGT_ID, smiles=smiles, scalars=raw_scalars)

def collect(root_dir: Path):
    # Setup paths and load labels
    data_dir = Path(__file__).parent.parent.parent / "data"
    df = pd.read_csv(data_dir / "split.csv")
    registry = load_registry()
    
    output_base = root_dir / "pocket_graphs"
    output_base.mkdir(parents=True, exist_ok=True)

    split_collections = {"train": [], "val": {"C1":[],"C2":[],"C3":[]}, "test": {"C1":[],"C2":[],"C3":[]}}
    config_dir_nonexistent = []
    no_smiles_found = []
    row_empty=[]
    no_config=[]
    no_best_path_found = []
    print("Building Atom-Level Pocket Graphs...")
    print(f"Check data on {data_dir}")
    for folder in boltz_dir.iterdir():
        if not folder.is_dir(): continue
        print(f"Process {folder}")
        folder_name = folder.name
        c = folder_name.split("config")[-1]
        config_dir = folder / "predictions" 
        if not config_dir.exists(): 
            config_dir_nonexistent.append(folder)
            continue
        # Match UGT_ID to labels
        UGT_ID,substrate_id = c.split("_")
        UGT_ID = int(UGT_ID)
        smiles = registry[substrate_id] if substrate_id in registry else None
        if smiles is None: 
            no_smiles_found.append(folder)
            continue
        row = df.loc[(df["UGT_ID"] == UGT_ID) & (df["SMILES_isomeric_1"] == smiles)]
        if row.empty: 
            print("row is empty")
            row_empty.append(folder)
            continue
            
        is_active = int(row["is_active"].values[0])
        split_name = row["split"].values[0]
        category = "val" if "val" in split_name.lower() else ("test" if "test" in split_name.lower() else "train")
        c_category = "C1" if "C1" in split_name else ("C2" if "C2" in split_name else "C3")
        
        config_subdirs = [p for p in config_dir.iterdir() if p.is_dir()]
        if not config_subdirs:
            no_config.append(config_dir)
            continue
        config_subdir = config_subdirs[0]
        cif_paths = [config_subdir / f"confidence_config{c}_model_{m}.json" for m in range(N_MODELS)]
        json_paths = []
        for m in range(N_MODELS):
            json_paths = json_paths + list(config_subdir.glob(f"*model_{m}.json"))
        best_json_path, interacting_chains = select_best_structure(json_paths)
        best_path = None
        if best_json_path is not None:
            model_str = best_json_path.stem.split("_")[-1]
            best_path = list(config_subdir.glob(f"*model_{model_str}.cif"))[0]
        else:
            no_best_path_found.append(config_subdir)
        if best_path is not None and best_path.exists():
            protein_seq = df.loc[df['UGT_ID']==UGT_ID,'prot_seq'].values[0]
            invalid_aas = set("XBUZJ")
            if any(aa in invalid_aas for aa in protein_seq):
                try:
                    n_feat = len(compute_all_features("C","A"))
                except Exception:
                    n_feat = 11
                feat_vec = [float("nan")]*n_feat
            else:
                feat_vec =  compute_all_features(smiles,protein_seq)
            raw_scalars = torch.tensor(
                feat_vec,
                dtype=torch.float
            )
            data = analyze_to_pocket_graph(best_path, is_active, c, model_str, UGT_ID, smiles, raw_scalars)
            if data:
                if category == "train":
                    split_collections[category].append(data)
                else:
                    split_collections[category][c_category].append(data)

    # Save splits
    for category, data_list in split_collections.items():
        if data_list:
            if category in ["val", "test"]:
                for c_category, graphs in data_list.items():
                    torch.save(graphs, output_base / f"{category}_{c_category}_pocket_dataset.pt")
                    print(f"Saved {len(graphs)} pocket graphs to {output_base} {category}_{c_category}_pocket_dataset.pt")
            else:

                torch.save(data_list, output_base / f"{category}_pocket_dataset.pt")
            print(f"Saved {len(data_list)} pocket graphs to {category}_pocket_dataset.pt")
    print(f"Amount of nonexistent directories : {len(config_dir_nonexistent)}")
    if len(config_dir_nonexistent)>0:
        print(config_dir_nonexistent[0])
    print(f"Amount of smiles not found: {len(no_smiles_found)}")
    if len(no_smiles_found)>0:
        print(no_smiles_found[0])
    print(f"Amount of empty rows : {len(row_empty)}")
    if len(row_empty)>0:
        print(row_empty[0])
    print(f"Amount of no config folder found : {len(no_config)}")
    if len(no_config)>0:
        print(no_config[0])
    print(f"Amount of times the best model was undeterminded : {len(no_best_path_found)}")
    if len(no_best_path_found)>0:
        print(no_best_path_found[0])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/boltz_results_root")
        sys.exit(1)
    collect(Path(sys.argv[1]).resolve())
