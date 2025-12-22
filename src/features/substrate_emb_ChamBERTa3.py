import argparse
from pathlib import Path
from typing import List, Dict, Any
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

# Global pooling setting for ChemBERTa-3 embeddings.
# Allowed values: "cls" or "mean"
POOLING = "cls"
MODEL_NAME = "DeepChem/ChemBERTa-100M-MLM"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = AutoModel.from_pretrained(MODEL_NAME, add_pooling_layer=False).to(device).eval()

def find_smiles_columns(df: pd.DataFrame) -> List[str]:
    """
    Try to automatically detect SMILES columns.

    Heuristic:
        - Any column whose name contains 'smiles' (case-insensitive)
    """
    smiles_cols = [
        col for col in df.columns
        if "smiles" in col.lower()
    ]
    if not smiles_cols:
        raise ValueError(
            "No SMILES columns found. Please ensure your CSV has columns like "
            "'SMILES_isomeric_1', 'SMILES_isomeric_2', etc."
        )
    return smiles_cols


def build_long_table(df: pd.DataFrame, smiles_cols: List[str]) -> pd.DataFrame:
    long_df = df.melt(
        id_vars=["substrate"],
        value_vars=smiles_cols,
        var_name="smiles_col",
        value_name="smiles",
    )

    # Fix: Convert to string but preserve NaN detection
    long_df["smiles"] = long_df["smiles"].astype(str).str.strip()

    def valid_smiles(x):
        if x is None:
            return False
        if x.lower() in ["", "nan", "none", "null"]:
            return False
        return True

    long_df = long_df[long_df["smiles"].apply(valid_smiles)]

    # Remove duplicates
    long_df = long_df.drop_duplicates(subset=["substrate", "smiles"]).reset_index(drop=True)

    return long_df



def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Mean pooling over tokens, masking out padding tokens.

    Args:
        token_embeddings: [batch_size, seq_len, hidden_dim]
        attention_mask:   [batch_size, seq_len]
    Returns:
        [batch_size, hidden_dim]
    """
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


def embed_smiles_batch(
    smiles_list: List[str],
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 128,
) -> torch.Tensor:
    """
    Compute ChemBERTa embeddings for a list of SMILES strings.

    The pooling method is controlled by the global POOLING variable:
        - "cls": use CLS token
        - "mean": use mean pooling over tokens
    """
    all_embeddings = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc=f"Encoding SMILES ({POOLING})"):
        batch_smiles = smiles_list[i: i + batch_size]

        encoded = TOKENIZER(
            batch_smiles,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = MODEL(**encoded)
            token_embeddings = outputs.last_hidden_state  # [B, L, H]

            if POOLING == "cls":
                batch_emb = token_embeddings[:, 0, :]               # CLS token
            elif POOLING == "mean":
                batch_emb = mean_pooling(token_embeddings, encoded["attention_mask"])
            else:
                raise ValueError(f"Invalid POOLING='{POOLING}'. Use 'cls' or 'mean'.")

        all_embeddings.append(batch_emb.cpu())

    return torch.cat(all_embeddings, dim=0)



def aggregate_by_substrate(
    long_df: pd.DataFrame,
    pair_embeddings: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Aggregate SMILES-level embeddings into one embedding per substrate.

    For substrates with multiple SMILES (e.g. racemic mixtures),
    we take the simple average of all their SMILES embeddings.

    Args:
        long_df: DataFrame with columns ['substrate', 'smiles'] (one row per SMILES).
        pair_embeddings: Tensor [num_pairs, hidden_dim], aligned with long_df rows.

    Returns:
        dict: substrate_name -> embedding tensor [hidden_dim]
    """
    if len(long_df) != pair_embeddings.size(0):
        raise ValueError("Number of rows in long_df does not match number of embeddings.")

    long_df = long_df.reset_index(drop=True)

    substrate_to_embedding: Dict[str, torch.Tensor] = {}

    # Group indices by substrate
    for substrate, group_idx in long_df.groupby("substrate").groups.items():
        idx_list = list(group_idx)
        emb = pair_embeddings[idx_list].mean(dim=0)
        substrate_to_embedding[substrate] = emb

    return substrate_to_embedding


def save_embeddings(
    substrate_to_embedding: Dict[str, torch.Tensor],
    original_df: pd.DataFrame,
    output_path: Path,
    model_name: str,
    smiles_source_file: Path,
    verbose=False
) -> None:
    """
    Save embeddings and metadata to a .pt file.

    We keep substrates in the order of the original CSV,
    but only include those for which we actually have an embedding.
    """
    seen = set()
    substrates = []
    emb_list = []

    for s in original_df["substrate"]:
        if s in substrate_to_embedding and s not in seen:
            substrates.append(s)
            emb_list.append(substrate_to_embedding[s])
            seen.add(s)

    if not emb_list:
        raise RuntimeError("No substrates had valid SMILES embeddings. Check your input CSV.")

    embeddings = torch.stack(emb_list, dim=0)  # [N, H]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "substrates": substrates,
        "embeddings": embeddings,
        "model_name": model_name,
        "smiles_source_file": str(smiles_source_file),
    }

    torch.save(payload, output_path)
    if verbose:
        print(f"   -> Saved {len(substrates)} substrate embeddings to: {output_path}")


def generate_CB3_emb(
    smiles_csv: str = None,
    output_path: str = None,
    verbose=False
):
    if smiles_csv is None:
        smiles_csv = f"{data_dir}/Substrate_SMILES.csv"
    smiles_csv_path = Path(smiles_csv)
    if output_path is None:
        output_path = f"{data_dir}/Substrate_Embeddings/ChemBERTa3_substrate_embeddings.pt"
    output_path = Path(output_path)

    if verbose:
        print(f"    Loading SMILES table from: {smiles_csv_path}")
    df = pd.read_csv(smiles_csv_path)

    smiles_cols = find_smiles_columns(df)
    if verbose:
        print(f"    Detected SMILES columns: {smiles_cols}")

    long_df = build_long_table(df, smiles_cols)
    if verbose:
        print(f"    Number of (substrate, SMILES) pairs: {len(long_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"    Using device: {device}")

    smiles_list = long_df["smiles"].tolist()
    pair_embeddings = embed_smiles_batch(
        smiles_list,
        device=device
    )

    substrate_to_embedding = aggregate_by_substrate(long_df, pair_embeddings)
    if verbose:
        print(f"    Number of unique substrates with embeddings: {len(substrate_to_embedding)}")

    save_embeddings(
        substrate_to_embedding=substrate_to_embedding,
        original_df=df,
        output_path=output_path,
        model_name=MODEL_NAME,
        smiles_source_file=smiles_csv_path,
        verbose=verbose
    )
    if verbose:
        print("    ChemBERTa substrate embeddings computed successfully.")