import torch
import re
import pandas as pd
import numpy as np
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
from torch.cuda.amp import autocast
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
torch.backends.cuda.enable_math_sdp(False)

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

def preprocess_protein_sequence(seq: str) -> str:
    """
    Replace ambiguous amino acids (U, Z, O, B) with X
    and insert spaces between each residue.
    """
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))

def load_prott5_model(device=None):
    """
    Load ProtT5 tokenizer and encoder model.
    Returns tokenizer, model, device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        do_lower_case=False,
        legacy=True
    )

    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc"
    ).to(device)

    if device.type == "cpu":
        model.to(torch.float32)

    return tokenizer, model, device
 
def compute_prott5_embeddings(sequences: list[str], tokenizer: T5Tokenizer, model: T5EncoderModel, device, batch_size=8):
    """
    Compute per-protein embeddings (mean-pooled)
    using ProtT5. Returns a torch.Tensor of shape (N, 1024).
    """
    processed = [preprocess_protein_sequence(s) for s in sequences]
    all_embeddings = []
    model.eval()

    for i in range(0, len(processed), batch_size):
        batch = processed[i : i + batch_size]
        ids = tokenizer(batch, add_special_tokens=True, padding=True, max_length=128, truncation=True)

        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            # outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, L, 1024)

        # mean pooling per sequence
        for j in range(hidden.size(0)):
            valid_len = attention_mask[j].sum().item()  # number of tokens incl. special
            # remove <cls> and <sep>
            residue_embeddings = hidden[j, 1 : valid_len - 1]
            emb = residue_embeddings.mean(dim=0)  # (1024,)
            all_embeddings.append(emb.cpu())

    return torch.stack(all_embeddings)

def generate_protein_emb(verbose=False):
    """
        Generate protein embeddings using the ProtT5 model.

        Steps:
            1. Load protein sequences from the UGT.csv file.
            2. Load the ProtT5 tokenizer and model.
            3. Compute embeddings for all sequences in batches.
            4. Save the embeddings to disk.

        Args:
            verbose (bool): If True, prints progress and information.
        """

    # Try to use full_dataset.csv if available, else fallback to UGT.csv
    import os
    full_dataset_path = data_dir / "full_dataset.csv"
    if full_dataset_path.exists():
        df = pd.read_csv(full_dataset_path)
        # Try to find the protein sequence column
        if "prot_seq" in df.columns:
            seqs = df["prot_seq"].tolist()
        else:
            # Try to find a column containing 'seq' (case-insensitive)
            seq_cols = [col for col in df.columns if "seq" in col.lower()]
            if seq_cols:
                seqs = df[seq_cols[0]].tolist()
                if verbose:
                    print(f"    Using protein sequences from column: {seq_cols[0]}")
            else:
                raise ValueError("No protein sequence column found. Expected 'prot_seq' or a column containing 'seq'.")
        # Save mapping of UGT_ID and UGT_Nomenclature for each embedding
        mapping_cols = []
        if 'UGT_ID' in df.columns:
            mapping_cols.append('UGT_ID')
        elif 'ugt_id' in df.columns:
            mapping_cols.append('ugt_id')
        if 'UGT_Nomenclature' in df.columns:
            mapping_cols.append('UGT_Nomenclature')
        elif 'ugt_nomenclature' in df.columns:
            mapping_cols.append('ugt_nomenclature')
        mapping_cols.append('prot_seq')
        protein_mapping = df[mapping_cols].copy()
    else:
        df = pd.read_csv(f"{data_dir}/UGT.csv")
        seqs = df["prot_seq"].tolist()
        protein_mapping = df[["UGT_ID", "UGT_Nomenclature", "prot_seq"]].copy()

    tokenizer, model, device = load_prott5_model()

    if verbose:
        print("    Computing protein embeddings in batches...")
    embeddings = compute_prott5_embeddings(seqs, tokenizer, model, device, batch_size=64)
    if verbose:
        print(f"    Computed embeddings with shape: {embeddings.shape}")

    output_dir = data_dir / "Protein_Embeddings"
    output_dir.mkdir(exist_ok=True)
    torch.save(embeddings, f"{output_dir}/embeddings.pt")

    # Save NumPy array
    embeddings_np = embeddings.detach().cpu().numpy()
    np.save(f"{output_dir}/protein_embeddings_prott5.npy", embeddings_np)

    # Save mapping CSV for robust downstream lookup
    protein_mapping.to_csv(f"{output_dir}/protein_embedding_mapping.csv", index=False)
    if verbose:
        print(f"    Saved protein embedding mapping to {output_dir}/protein_embedding_mapping.csv")


if __name__ == "__main__":
    generate_protein_emb(verbose=True)