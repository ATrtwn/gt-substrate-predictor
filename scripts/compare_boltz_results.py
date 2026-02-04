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


import http.server
import socketserver
import webbrowser
import os
import io
def visualize_comparison(exp_file, exp_binding, pred_file, pred_binding, ligands, metrics):
    # --- 1. SETUP EXPERIMENTAL VIEW ---
    view_exp = py3Dmol.view(width=500, height=500)
    with open(str(exp_file), "r") as f:
        view_exp.addModel(f.read(), "cif")
    view_exp.setStyle({'model': -1}, {'cartoon': {'color': 'lightgrey'}})
    for res in exp_binding:
        view_exp.addStyle({'chain': res.get_parent().id, 'resi': res.id[1]}, {'cartoon': {'color': 'red'}})
    for lig in ligands: # Assuming ligands are similar
        view_exp.addStyle({'resn': lig.get_resname()}, {'stick': {'colorscheme': 'greenCarbon'}})
    view_exp.zoomTo()

    # --- 2. SETUP PREDICTION VIEW ---
    view_pred = py3Dmol.view(width=500, height=500)
    with open(str(pred_file), "r") as f:
        view_pred.addModel(f.read(), "cif")
    view_pred.setStyle({'model': -1}, {'cartoon': {'color': 'lightgrey'}})
    for res in pred_binding:
        view_pred.addStyle({'chain': res.get_parent().id, 'resi': res.id[1]}, {'cartoon': {'color': 'red'}})
    for lig in ligands:
        view_pred.addStyle({'resn': lig.get_resname()}, {'stick': {'colorscheme': 'greenCarbon'}})
    view_pred.zoomTo()

    # --- 3. INJECT INTO A TABLE ---
    html_content = f"""
    <html>
        <head>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
            <style>
                body {{ background-color: #252526; color: white; font-family: monospace; text-align: center; }}
                .container {{ display: flex; justify-content: center; gap: 20px; padding: 20px; }}
                .box {{ background: #333; padding: 10px; border-radius: 8px; }}
                .stats {{ background: #1e1e1e; padding: 20px; border: 1px solid #444; width: 1000px; margin: 20px auto; text-align: left; }}
            </style>
        </head>
        <body>
            <h2>Structural Comparison</h2>
            <div class="container">
                <div class="box">
                    <div>EXPERIMENTAL REFERENCE</div>
                    {view_exp._make_html()}
                </div>
                <div class="box">
                    <div>AI PREDICTION</div>
                    {view_pred._make_html()}
                </div>
            </div>
            
            <div class="stats">
                <b style="color:#569cd6;">GLOBAL FOLD:</b> {metrics['rmsd_global']:.2f} Å ({metrics['fold_status']})<br>
                <b style="color:#4ec9b0;">BINDING SITE RMSD:</b> {metrics['rmsd_binding']:.2f} Å<br>
                <b style="color:#ce9178;">SITE ACCURACY:</b> {metrics['conserved_pct']:.1f}% ({metrics['site_status']})
            </div>
        </body>
    </html>
    """

    # --- 4. SERVER LOGIC ---
    class MemoryHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
        def log_message(self, format, *args): return

    with socketserver.TCPServer(("", 0), MemoryHandler) as httpd:
        port = httpd.socket.getsockname()[1]
        url = f"http://localhost:{port}"
        print(f"Comparison ready at {url}")
        webbrowser.open(url)
        httpd.handle_request()
def visualize_structure_exp(cif_file, protein_residues, binding_site_residues, conserved_residues, ligands, output_html):
    # 1. Create the view and load content
    view = py3Dmol.view(width=800, height=600)
    with open(str(cif_file), "r") as f:
        pdb_text = f.read()
    view.addModel(pdb_text, "cif")

    # 1. Global Protein Style: The "Spirals" (Cartoon)
    # We set a base color for the whole protein
    view.setStyle({'model': -1}, {'cartoon': {'color': 'lightgrey'}})

    # 2. Binding site residues: Colored "Spirals"
    # Instead of 'stick', we use 'cartoon' to keep the spiral shape but change the color
    for res in binding_site_residues:
        view.addStyle({'chain': res.get_parent().id, 'resi': res.id[1]}, 
                      {'cartoon': {'color': 'red'}})

    # 3. Conserved residues: Colored "Spirals"
    if conserved_residues is not None:
        for res in conserved_residues:
            view.addStyle({'chain': res.get_parent().id, 'resi': res.id[1]}, 
                          {'cartoon': {'color': 'blue'}})

    # 4. Ligands: Keep these as sticks! 
    # (Ligands don't have secondary structure, so they can't be spirals)
    for ligand in ligands:
        view.addStyle({'chain': ligand.get_parent().id, 'resi': ligand.id[1]}, 
                      {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})

    # 5. Focus and Zoom
    view.setStyle({'chain': ['B', 'C', 'D']}, {}) # Hide other chains if necessary
    view.zoomTo()
    # --- DIRECT TO BROWSER LOGIC ---
    html_content = f"""
    <html>
        <head>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        </head>
        <body style="margin:0; padding:0; background-color: #252526;">
            <div style="background:#333; color:white; padding:10px; font-family: sans-serif;">
                <b>EXPERIMENTAL REFERENCE:</b> {os.path.basename(cif_file)}
            </div>
            {view._make_html()}
        </body>
    </html>
    """

    class MemoryHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
        def log_message(self, format, *args): return 

    with socketserver.TCPServer(("", 0), MemoryHandler) as httpd:
        port = httpd.socket.getsockname()[1]
        url = f"http://localhost:{port}"
        print(f"Opening experimental viewer at {url}")
        webbrowser.open(url)
        httpd.handle_request()

def visualize_structure(cif_file, binding_site_residues, conserved_residues, ligands, 
                        rmsd_global, rmsd_binding, num_conserved, exp_binding_residues_len, 
                        conserved_pct, fold_status, site_status):
    
    # --- 1. 3D VIEW SETUP (Spirals/Cartoon) ---
    view = py3Dmol.view(width=800, height=600)
    with open(str(cif_file), "r") as f:
        pdb_text = f.read()
    view.addModel(pdb_text, "cif")

    # 1. Global Protein Style: The "Spirals" (Cartoon)
    # We set a base color for the whole protein
    view.setStyle({'model': -1}, {'cartoon': {'color': 'lightgrey'}})

    # 2. Binding site residues: Colored "Spirals"
    # Instead of 'stick', we use 'cartoon' to keep the spiral shape but change the color
    for res in binding_site_residues:
        view.addStyle({'chain': res.get_parent().id, 'resi': res.id[1]}, 
                      {'cartoon': {'color': 'red'}})

    # 3. Conserved residues: Colored "Spirals"
    if conserved_residues is not None:
        for res in conserved_residues:
            view.addStyle({'chain': res.get_parent().id, 'resi': res.id[1]}, 
                          {'cartoon': {'color': 'blue'}})

    # 4. Ligands: Keep these as sticks! 
    # (Ligands don't have secondary structure, so they can't be spirals)
    for ligand in ligands:
        view.addStyle({'chain': ligand.get_parent().id, 'resi': ligand.id[1]}, 
                      {'stick': {'colorscheme': 'greenCarbon', 'radius': 0.3}})

    # 5. Focus and Zoom
    view.setStyle({'chain': ['B', 'C', 'D']}, {}) # Hide other chains if necessary
    view.zoomTo()
    print(f"Debug:RSMD Global: {rmsd_global}, RMSD Binding: {rmsd_binding}, Conserved: {num_conserved}/{exp_binding_residues_len}")
    # --- 2. FORMATTED HTML OUTPUT ---
    # This creates the "box" under the picture
    metrics_html = f"""
    <div style="font-family: 'Courier New', Courier, monospace; 
                background-color: #1e1e1e; color: #d4d4d4; 
                padding: 20px; margin-top: 10px; border-radius: 5px; 
                width: 760px; line-height: 1.5; border: 1px solid #333;">
        <div style="color: #569cd6; font-weight: bold; border-bottom: 1px solid #333; padding-bottom: 5px; margin-bottom: 10px;">
            {'='*15} MODEL ANALYSIS {'='*15}
        </div>
        <div style="margin-bottom: 5px;"><b style="color: #ce9178;">FILE:</b> {os.path.basename(cif_file)}</div>
        <hr style="border: 0; border-top: 1px solid #333;">
        
        <div style="margin-top: 10px;">
            <b style="color: #4ec9b0;">Protein Backbone (Global):</b><br>
            &nbsp;&nbsp;- RMSD: {rmsd_global:.2f} Å ({fold_status})
        </div>
        
        <div style="margin-top: 10px;">
            <b style="color: #4ec9b0;">Binding Site (Local):</b><br>
            &nbsp;&nbsp;- RMSD: {rmsd_binding:.2f} Å<br>
            &nbsp;&nbsp;- Conserved Residues: {num_conserved} of {exp_binding_residues_len} ({conserved_pct:.1f}%)<br>
            &nbsp;&nbsp;- Summary: <span style="color: {'#4fc1ff' if site_status == 'High Accuracy' else '#dcdcaa'};">{site_status}</span>
        </div>
        <div style="color: #569cd6; font-weight: bold; margin-top: 10px;">{'='*46}</div>
    </div>
    """

    html_content = f"""
    <html>
        <head>
            <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        </head>
        <body style="margin:0; padding:20px; background:#252526; display: flex; flex-direction: column; align-items: center;">
            <div style="background:#333; color:white; padding:10px; width:800px; text-align:center; border-radius: 5px 5px 0 0;">
                Visualizing: {os.path.basename(cif_file)}
            </div>
            {view._make_html()}
            {metrics_html}
        </body>
    </html>
    """

    # --- 3. SERVER LOGIC ---
    # (Same as your existing code to serve the one-time request)
    class MemoryHandler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(html_content.encode("utf-8"))
        def log_message(self, format, *args): return

    with socketserver.TCPServer(("", 0), MemoryHandler) as httpd:
        port = httpd.socket.getsockname()[1]
        url = f"http://localhost:{port}"
        print(f"Opening viewer at {url}")
        webbrowser.open(url)
        httpd.handle_request()
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
target_chain_id = 'A' 
if target_chain_id not in chains_data:
    target_chain_id = list(chains_data.keys())[0]

exp_data = chains_data[target_chain_id]
exp_binding_residues = get_binding_site(
    exp_data["atoms"], 
    exp_data["residues"], 
    exp_data["ligands"]
)

# Visualize the Experimental reference first
visualize_structure_exp(
    experimental_cif, 
    exp_data["residues"], 
    exp_binding_residues, 
    None, 
    exp_data["ligands"], 
    f"experimental_ref_{target_chain_id}.html",

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
     # --- VISUALIZE PREDICTION ---
    # This shows the predicted protein with its own binding site and highlights 
    #the residues that matched the experimental distance cutoff in blue.
    visualize_structure(
        cif_file=pred_file,
        binding_site_residues=pred_binding_residues,
        conserved_residues=conserved_res,
        ligands=pred_ligands,
        rmsd_global=rmsd_global,
        rmsd_binding=rmsd_binding,
        num_conserved=num_conserved,
        exp_binding_residues_len=len(exp_binding_residues), # Passing the length specifically
        conserved_pct=conserved_pct,
        fold_status=fold_status,
        site_status=site_status
    )
    
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
    break
print("Analysis complete.")

print("Analysis complete. Summary:")
for res in results:
    print(res)