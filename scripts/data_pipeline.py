import warnings
warnings.filterwarnings(
    "ignore",
    message="TypedStorage is deprecated"
)
from src.data.preprocessing import prepare_dataset
from src.data.data_split import split_and_analyse_dataset
from src.data.embeddings import create_embeddings

def main():
    """
        Run the full end-to-end pipeline for dataset preparation, splitting, and embedding generation.

        1. Dataset preprocessing:
           - Filters and standardizes raw input data
           - Prepares auxiliary files (e.g. FASTA, KPGT inputs)
           - Merges all sources into a unified dataset

        2. Dataset splitting and analysis:
           - Splits protein–substrate pairs into train, validation, and test sets
           - Enforces C1 / C2 / C3 generalization constraints
           - Computes and reports dataset statistics

        3. Embedding generation:
           - Generates protein and substrate embeddings
           - Optionally concatenates embeddings

        The pipeline is intended to be run as a single entry point for reproducible
        data preparation and feature generation.
    """

    print("\n==== [1/3] Preprocessing dataset ====")
    df = prepare_dataset(verbose=False)

    print("\n==== [2/3] Splitting into C1/C2/C3 ====")
    df_all = split_and_analyse_dataset(df, plots=False, verbose=False)

    print("\n==== [3/3] Generating embeddings ====")
    create_embeddings(embeddings_to_generate='all', concat=True, verbose=false)

if __name__ == "__main__":
    main()