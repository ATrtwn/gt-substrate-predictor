from src.features.protein_emb import generate_protein_emb
from src.features.KPGT_emb import generate_KPGT_emb
from src.features.substrate_emb_ChamBERTa2 import generate_CB2_emb
from src.features.substrate_emb_ChamBERTa3 import generate_CB3_emb
from src.features.concatenate_embeddings import concatenate_embeddings
from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

def create_embeddings(
    embeddings_to_generate = None,
    concat: bool = True,
    verbose: bool = False
):
    """
    Generate protein and/or substrate embeddings and optionally concatenate them.

    This function orchestrates the embedding pipeline by:
    1. Generating selected embeddings (protein and/or substrate)
    2. Optionally concatenating embeddings

    Args:
        embeddings_to_generate (list[str] or str, optional):
            Which embeddings to generate.
            Valid options are:
                - 'protein'
                - 'kpgt'
                - 'chemberta2'
                - 'chemberta3'
            Use 'all' or None to generate all available embeddings.
        concat (bool):
            If True, concatenates the generated embeddings into combined datasets.
    """
    # default = all embeddings
    all_embeddings = {
        'protein': generate_protein_emb,
        'kpgt': generate_KPGT_emb,
        'chemberta2': generate_CB2_emb,
        'chemberta3': generate_CB3_emb
    }

    if embeddings_to_generate is None or embeddings_to_generate == "all":
        embeddings_to_generate = list(all_embeddings.keys())

    # check that requested embeddings are valid
    invalid = set(embeddings_to_generate) - set(all_embeddings.keys())
    if invalid:
        raise ValueError(f"Invalid embeddings requested: {invalid}. Valid options: {list(all_embeddings.keys())}")

    print("\n== [1/2] Embedding generation ==")

    # generate requested embeddings
    for emb_name in embeddings_to_generate:
        if verbose:
            print(f" - Generating {emb_name} embeddings")
        all_embeddings[emb_name](verbose=verbose)

    # concatenate if requested
    if concat:
        print("== [2/2] Concatenating substrate embeddings ==")
        # Only concatenate embeddings that were generated
        substrate_embeddings = {
            'kpgt': generate_KPGT_emb,
            'chemberta2': generate_CB2_emb,
            'chemberta3': generate_CB3_emb
        }
        embeddings_for_concat = [emb for emb in embeddings_to_generate if emb in substrate_embeddings]
        if set(embeddings_for_concat) == set(substrate_embeddings):
            embeddings_for_concat = "all"
        concatenate_embeddings(embeddings=embeddings_for_concat, verbose=verbose)
    else:
        print("== Skipping embedding concatenation ==")
