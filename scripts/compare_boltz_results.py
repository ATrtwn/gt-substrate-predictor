import os
from pathlib import Path
import numpy as np
import py3Dmol
from Bio.PDB import MMCIFParser, is_aa, NeighborSearch, Superimposer
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB.Residue import Residue
from Bio.PDB.Chain import Chain

boltz_path = Path(__file__).parent.parent / "boltz_test" / "boltz_results_config" /"predictions" / "config"

# --- CONFIGURATION ---
experimental_cif = Path(__file__).parent.parent / "9J9K.cif"  # Path to experimental structure
predicted_files = [boltz_path /"config_model_0.cif", boltz_path /"config_model_1.cif", boltz_path /"config_model_2.cif", boltz_path /"config_model_3.cif", boltz_path /"config_model_4.cif"]
binding_site_cutoff = 6.0            # Å

binding_site_cutoff = 6.0            # Å for binding site
conserved_cutoff = 2.0

# --- UTILITY FUNCTIONS ---

def load_structure(cif_file: str):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file)
    protein_atoms, protein_residues, ligands = [], [], []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue, standard=True):
                    protein_atoms.extend(list(residue.get_atoms()))
                    protein_residues.append(residue)
                else:
                    if residue.get_resname() not in ["HOH", "WAT"]:
                        ligands.append(residue)
    return protein_atoms, protein_residues, ligands

def get_binding_site(protein_atoms, protein_residues, ligands, cutoff=binding_site_cutoff):
    ns = NeighborSearch(protein_atoms)
    binding_site_residues = set()
    for ligand in ligands:
        for atom in ligand.get_atoms():
            neighbors = ns.search(atom.coord, cutoff, level="R")
            for res in neighbors:
                binding_site_residues.add(res)
    return binding_site_residues



def calculate_rmsd(ref_residues, target_residues):
    # Sort both by their actual position in the structure
    ref_res = sorted([res for res in ref_residues if 'CA' in res], key=lambda x: x.id[1])
    tar_res = sorted([res for res in target_residues if 'CA' in res], key=lambda x: x.id[1])

    # Ensure we only compare up to the shortest sequence length available
    min_len = min(len(ref_res), len(tar_res))
    if min_len == 0:
        return None

    ref_atoms = [ref_res[i]['CA'] for i in range(min_len)]
    tar_atoms = [tar_res[i]['CA'] for i in range(min_len)]

    sup = Superimposer()
    sup.set_atoms(ref_atoms, tar_atoms)
    sup.apply(tar_atoms)
    return sup.rms

def count_conserved_by_distance(exp_binding_site, pred_binding_site, cutoff=conserved_cutoff):
    """
    Matches residues by their relative position in the binding site list 
    since residue numbers may differ between files.
    """
    # Get CA atoms and sort to maintain relative order
    exp_ca = sorted([res for res in exp_binding_site if 'CA' in res], key=lambda x: x.id[1])
    pred_ca = sorted([res for res in pred_binding_site if 'CA' in res], key=lambda x: x.id[1])

    if not exp_ca or not pred_ca:
        return 0, set()

    # Use NeighborSearch on all predicted CA atoms
    ns = NeighborSearch([res['CA'] for res in pred_ca])
    conserved = set()

    for res in exp_ca:
        # If any predicted CA is within cutoff distance of this experimental CA
        neighbors = ns.search(res['CA'].coord, cutoff, level="R")
        if neighbors:
            conserved.add(res)
            
    return len(conserved), conserved

def visualize_structure(cif_file, protein_residues, binding_site_residues, conserved_residues, ligands, output_html):
    view = py3Dmol.view(width=800, height=600)
    with open(str(cif_file), "r") as f:
        pdb_text = f.read()
    view.addModel(pdb_text, "cif")
    # Protein cartoon
    view.setStyle({'chain':'A'}, {'cartoon':{'color':'lightgrey'}})
    # Binding site residues red
    for res in binding_site_residues:
        chain_id = res.get_parent().id
        resnum = res.id[1]
        view.setStyle({'chain': chain_id, 'resi': resnum}, {'stick': {'color':'red'}})
    # Conserved residues blue
    if conserved_residues is not None:
        for res in conserved_residues:
            chain_id = res.get_parent().id
            resnum = res.id[1]
            view.setStyle({'chain': chain_id, 'resi': resnum}, {'stick': {'color':'blue'}})
    # Ligands green
    for ligand in ligands:
        chain_id = ligand.get_parent().id
        resnum = ligand.id[1]
        view.setStyle({'chain': chain_id, 'resi': resnum}, {'stick': {'color':'green'}})
    view.zoomTo()
    view.write_html(output_html)
    os.startfile(output_html)
from collections import defaultdict

def split_by_chain(atoms, residues, ligands):
    chains = defaultdict(lambda: {
        "atoms": [],
        "residues": [],
        "ligands": []
    })

    for atom in atoms:
        chain_id = atom.get_parent().get_parent().id
        chains[chain_id]["atoms"].append(atom)

    for res in residues:
        chain_id = res.get_parent().id
        chains[chain_id]["residues"].append(res)

    for lig in ligands:
        chain_id = lig.get_parent().id
        chains[chain_id]["ligands"].append(lig)

    return dict(chains)
# --- MAIN ANALYSIS ---

# 1. Load and Split Experimental
exp_atoms, exp_residues, exp_ligands = load_structure(experimental_cif)
chains_data = split_by_chain(exp_atoms, exp_residues, exp_ligands)

# 2. Select specific experimental chain (e.g., Chain A)
target_chain_id = 'B' 
if target_chain_id not in chains_data:
    target_chain_id = list(chains_data.keys())[0]

exp_data = chains_data[target_chain_id]
exp_binding_residues = get_binding_site(
    exp_data["atoms"], 
    exp_data["residues"], 
    exp_data["ligands"]
)

# Visualize the Experimental reference first
visualize_structure(
    experimental_cif, 
    exp_data["residues"], 
    exp_binding_residues, 
    None, 
    exp_data["ligands"], 
    f"experimental_ref_{target_chain_id}.html"
)

results = []

# 3. Process Predictions
for pred_file in predicted_files:
    pred_atoms, pred_residues, pred_ligands = load_structure(pred_file)
    pred_binding_residues = get_binding_site(pred_atoms, pred_residues, pred_ligands)
    
    # Calculations
    rmsd_global = calculate_rmsd(exp_data["residues"], pred_residues)
    rmsd_binding = calculate_rmsd(list(exp_binding_residues), list(pred_binding_residues))
    num_conserved, conserved_res = count_conserved_by_distance(exp_binding_residues, pred_binding_residues)
    
    results.append((pred_file.name, rmsd_global, rmsd_binding, num_conserved))
    
    # --- VISUALIZE PREDICTION ---
    # This shows the predicted protein with its own binding site and highlights 
    # the residues that matched the experimental distance cutoff in blue.
    visualize_structure(
        pred_file, 
        pred_residues, 
        pred_binding_residues, 
        conserved_res, 
        pred_ligands, 
        f"viz_{pred_file.stem}.html"
    )
    
    # --- CALCULATE METRICS ---
    conserved_pct = (num_conserved / len(exp_binding_residues)) * 100 if exp_binding_residues else 0

    # Determine structural quality status
    if rmsd_global and rmsd_global < 2.0:
        fold_status = "Correct Fold"
    elif rmsd_global and rmsd_global < 4.0:
        fold_status = "Approximate Fold"
    else:
        fold_status = "Poor Fold"

    # Determine binding site accuracy
    site_status = "High Accuracy" if conserved_pct > 70 else "Moderate" if conserved_pct > 30 else "Low Accuracy"

    # --- FORMATTED OUTPUT ---
    print(f"{'='*40}")
    print(f"MODEL: {pred_file.name}")
    print(f"{'-'*40}")
    print(f"Protein Backbone (Global):")
    print(f"  - RMSD: {rmsd_global:.2f} Å ({fold_status})")
    print(f"Binding Site (Local):")
    print(f"  - RMSD: {rmsd_binding:.2f} Å")
    print(f"  - Conserved Residues: {num_conserved} of {len(exp_binding_residues)} ({conserved_pct:.1f}%)")
    print(f"  - Summary: {site_status}")
    print(f"{'='*40}\n")

print("Analysis complete.")

print("Analysis complete. Summary:")
for res in results:
    print(res)
