from src.features.KPGT_emb import generate_KPGT_emb
from src.features.substrate_emb_ChamBERTa2 import generate_CB2_emb
from src.features.substrate_emb_ChamBERTa3 import generate_CB3_emb
from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent / "data"

def create_embeddings():
    """embeddings."""
    generate_KPGT_emb()
    generate_CB2_emb()
    generate_CB3_emb()
    pass