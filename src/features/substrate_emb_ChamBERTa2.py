from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
import pubchempy as pcp
import time
from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

# Load ChamBERTA2 model and tokenizer
model_name = "DeepChem/ChemBERTa-77M-MTR"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
model.eval()

# Convert to molecules in SMILES format
def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)

# Lookup SMILES from PubChem using substrate name
def get_smiles_from_pubchem(substrate_name):
    """Query PubChem to get SMILES from chemical name."""
    try:
        compounds = pcp.get_compounds(substrate_name, 'name')
        if compounds:
            return compounds[0].isomeric_smiles  # Use isomeric SMILES for stereochemistry
        return None
    except Exception as e:
        print(f"Error looking up {substrate_name}: {e}")
        return None

# Generate embeddings for a list of SMILES strings
def generate_embeddings(smiles_list):
    embeddings = []
    for smiles in smiles_list:
        if pd.isna(smiles) or smiles == "":
            embeddings.append(None)
            continue
        mol = smiles_to_mol(smiles)
        if mol is None:
            embeddings.append(None)
            continue
        input_ids = tokenizer(smiles, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            outputs = model(input_ids)
        # Use the [CLS] token representation as the embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)
    return embeddings

def generate_CB2_emb(
        csv_path: str = None,
        output_path: str = None,
        verbose=False):
    """Load substrate CSV with SMILES, generate embeddings, and save."""
    # Process substrate data
    if csv_path is None:
        csv_path = data_dir / "Substrate_SMILES.csv"

    # Load data
    df = pd.read_csv(csv_path)
    if verbose:
        print(f"    Loaded {len(df)} substrates from {csv_path}")
    
    # Use SMILES from CSV (SMILES_isomeric_1 column)
    if 'SMILES_isomeric_1' in df.columns:
        df['smiles'] = df['SMILES_isomeric_1']
        if verbose:
            print(f"    Using SMILES from SMILES_isomeric_1 column")
    elif 'smiles' in df.columns:
        if verbose:
            print(f"    Using existing smiles column")
    else:
        raise ValueError("No SMILES column found. Expected 'SMILES_isomeric_1' or 'smiles'.")

    if verbose:
        print(f"    Found SMILES for {df['smiles'].notna().sum()} substrates")
    
    # Generate embeddings
    if verbose:
        print("    Generating ChemBERTa-2 embeddings...")
    embeddings = generate_embeddings(df['smiles'].tolist())
    
    # Add embeddings to dataframe
    df['embedding'] = embeddings
    
    # Save results
    if output_path is None:
        output_dir = data_dir / "Substrate_Embeddings"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "Substrate_with_embeddings_chemberta2.csv"
    
    # Save CSV with SMILES
    df.to_csv(output_path, index=False)
    if verbose:
        print(f"   -> Saved results to {output_path}")
    
    # Save embeddings as numpy array (for ML)
    valid_embeddings = [emb for emb in embeddings if emb is not None]
    if valid_embeddings:
        emb_array = np.array(valid_embeddings)
        output_dir = data_dir / "Substrate_Embeddings"
        output_dir.mkdir(exist_ok=True)
        npy_path = output_dir / "substrate_embeddings_chemberta2.npy"
        np.save(npy_path, emb_array)
        if verbose:
            print(f"   -> Saved {emb_array.shape} embeddings to {npy_path}")