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
            continue  # skip invalid
        mol = smiles_to_mol(smiles)
        if mol is None:
            continue  # skip invalid
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
    

    # Support for full_dataset.csv: try to find the right SMILES column
    if 'SMILES_isomeric_1' in df.columns:
        df['smiles'] = df['SMILES_isomeric_1']
        if verbose:
            print(f"    Using SMILES from SMILES_isomeric_1 column")
    elif 'smiles' in df.columns:
        if verbose:
            print(f"    Using existing smiles column")
    # Try to find a likely SMILES column in full_dataset.csv
    else:
        # Try to find a column containing 'smiles' (case-insensitive)
        smiles_cols = [col for col in df.columns if 'smiles' in col.lower()]
        if smiles_cols:
            df['smiles'] = df[smiles_cols[0]]
            if verbose:
                print(f"    Using SMILES from column: {smiles_cols[0]}")
        else:
            raise ValueError("No SMILES column found. Expected 'SMILES_isomeric_1', 'smiles', or a column containing 'smiles'.")

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
    valid_substrates = df.loc[[emb is not None for emb in embeddings], 'substrate']
    if valid_embeddings:
        emb_array = np.array(valid_embeddings)
        output_dir = data_dir / "Substrate_Embeddings"
        output_dir.mkdir(exist_ok=True)
        npy_path = output_dir / "substrate_embeddings_chemberta2.npy"
        np.save(npy_path, emb_array)
        # Save substrate list in embedding order
        csv_path = output_dir / "substrate_list_chemberta2.csv"
        valid_substrates.to_csv(csv_path, index=False, header=True)
        if verbose:
            print(f"   -> Saved {emb_array.shape} embeddings to {npy_path}")
            print(f"   -> Saved substrate list to {csv_path}")


if __name__ == "__main__":
    generate_CB2_emb(
        csv_path=None,  # Use default: full_dataset.csv or Substrate_SMILES.csv
        output_path=None,  # Use default output path
        verbose=True
    )