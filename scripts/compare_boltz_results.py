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



# def get_chain_with_ligand(atom_list):
#     """
#     Process a list of Atom objects and separate protein residues and ligands.
#     Returns: (protein_residues, ligands)
#     """
#     if not all(hasattr(atom, "get_parent") for atom in atom_list):
#         raise TypeError("Input must be a list of Atom objects")

#     # Collect residues from atoms, deduplicate by (chain_id, res_id)
#     res_dict = {}
#     for atom in atom_list:
#         res = atom.get_parent()        # Residue object
#         chain_id = res.get_parent().id # chain ID
#         key = (chain_id, res.get_id()) # unique key
#         res_dict[key] = res

#     residues = list(res_dict.values())

#     # Separate protein residues
#     protein_residues = [res for res in residues if is_aa(res, standard=True)]

#     # Separate ligands (non-protein, excluding water)
#     ligands = [res for res in residues
#                if not is_aa(res, standard=True) and res.get_resname() not in ["HOH", "WAT"]]

#     if ligands:
#         return protein_residues, ligands
#     else:
#         raise ValueError("No ligand found in the atom list.")


def calculate_rmsd(ref_residues, target_residues):
    # Match residues by (chain, resseq) for global RMSD
    ref_dict = {(res.get_parent().id, res.id[1]): res for res in ref_residues if 'CA' in res}
    target_dict = {(res.get_parent().id, res.id[1]): res for res in target_residues if 'CA' in res}
    common_keys = set(ref_dict.keys()) & set(target_dict.keys())
    if not common_keys:
        return None
    ref_atoms = [ref_dict[k]['CA'] for k in common_keys]
    target_atoms = [target_dict[k]['CA'] for k in common_keys]
    sup = Superimposer()
    sup.set_atoms(ref_atoms, target_atoms)
    sup.apply(target_atoms)
    return sup.rms

def count_conserved_by_distance(exp_binding_site, pred_binding_site, cutoff=conserved_cutoff):
    """
    Counts residues in the experimental binding site that have a predicted residue
    within cutoff distance of their Cα atom.
    """
    pred_atoms = [res['CA'] for res in pred_binding_site if 'CA' in res]
    print(pred_atoms)
    print([res['CA'] for res in exp_binding_site if 'CA' in res])
    ns = NeighborSearch(pred_atoms)
    conserved = set()
    for res in exp_binding_site:
        if 'CA' not in res:
            continue
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

exp_atoms, exp_residues, exp_ligands = load_structure(experimental_cif)
chains = split_by_chain(exp_atoms, exp_residues, exp_ligands)

for chain_id, data in chains.items():
    binding = get_binding_site(
        data["atoms"],
        data["residues"],
        data["ligands"]
    )
    print(f"Chain {chain_id}: {len(binding)} binding residues")

    print(f"Experimental structure has {len(data['ligands'])} ligand(s) and {len(binding)} binding site residues.\n")
    visualize_structure(experimental_cif, exp_residues, binding, None, exp_ligands, f"{experimental_cif}_{chain_id}_visualization.html")

results = []

for pred_file in predicted_files:
    pred_atoms, pred_residues, pred_ligands = load_structure(pred_file)
    pred_binding_residues = get_binding_site(pred_atoms, pred_residues, pred_ligands)
    
    # RMSD calculations
    rmsd_global = calculate_rmsd(exp_residues, pred_residues)
    rmsd_binding = calculate_rmsd(list(exp_binding_residues), list(pred_binding_residues))
    
    # Conserved residues by distance
    num_conserved, conserved_residues = count_conserved_by_distance(exp_binding_residues, pred_binding_residues)
    
    results.append((pred_file, rmsd_global, rmsd_binding, num_conserved))
    
    print(f"--- {pred_file} ---")
    print(f"Global RMSD: {rmsd_global:.2f} Å" if rmsd_global else "Global RMSD: N/A")
    print(f"Binding site RMSD: {rmsd_binding:.2f} Å" if rmsd_binding else "Binding site RMSD: N/A")
    print(f"Conserved binding-site residues: {num_conserved} / {len(exp_binding_residues)}\n")
    
    visualize_structure(pred_file, pred_residues, pred_binding_residues, conserved_residues, pred_ligands, f"{pred_file}_visualization.html")

print("Analysis complete. Summary:")
for res in results:
    print(res)
