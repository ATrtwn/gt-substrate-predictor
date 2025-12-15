
import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import requests
from tqdm import tqdm  # progress bar

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

def create_original_dataset(verbose=False):
    all_exist = all(os.path.exists(os.path.join(data_dir, f)) for f in ["UGT.csv", "Activity.csv", "Substrate_SMILES.csv"])

    if all_exist:
        if verbose:
            print(" -  All CSV files already exist. Skipping create_csv()")
    else:
        if verbose:
            print(" -  Some CSV files are missing. Running create_csv()...")
            print("    ⚠️ Please make sure you have a .env file in the project root containing:")
            print("   ACCESS_DB_PATH=/full/path/to/your/database.accdb")
            print("   ACCESS_DB_PASSWORD=yourpassword")
        create_csv(verbose=verbose)

    activity = pd.read_csv(os.path.join(data_dir, "Activity.csv"))
    UGT = pd.read_csv(os.path.join(data_dir, "UGT.csv"))
    substrate = pd.read_csv(os.path.join(data_dir, "Substrate_SMILES.csv"))
    if verbose:
        print(f" - Data overview: ")
        print(f"    activity shape: {activity.shape}")
        print(f"    UGT shape: {UGT.shape}")
        print(f"    substrate shape: {substrate.shape}")

    df_merged = activity.merge(UGT, left_on="UGT_trivial_name", right_on="UGT_trivial_name", how="left")
    df_merged = df_merged.merge(substrate, left_on="substrate", right_on="substrate", how="left")
    df_merged.to_csv(os.path.join(data_dir, "merged.csv"), index=False)

def create_full_dataset(verbose=False):
    ### original gt/substrate dataset
    df_original = pd.read_csv(os.path.join(data_dir, "merged.csv"))
    # binarize
    df_original = binarize_activity(df_original)
    df_original = df_original.sort_values('is_active', ascending=False)
    df_original = df_original.drop_duplicates(subset=['UGT_ID', 'substrate'], keep='first')
    df_original = df_original[['UGT_ID', 'substrate', 'UGT_Nomenclature',
                               'nt_seq', 'prot_seq', 'molecule', 'SMILES_isomeric_1',
                               'is_active']].drop_duplicates()
    df_original['dataset'] = 'original'
    if verbose:
        print(f" - original dataset size: {len(df_original)}")
        print(f"    columns: {df_original.columns}")

    ### ESP dataset
    if verbose:
        print(" - Get data from ESP...")
    df_new_ESP = pd.read_csv(os.path.join(data_dir, "data_ESP.csv"))
    df_new_ESP = df_new_ESP.rename(columns={
        "MolecularFormula": "molecule",
        "ConnectivitySMILES": "SMILES_isomeric_1"
    })
    if verbose:
        print(f" - ESP dataset size: {len(df_new_ESP)}")
        print(f"    columns: {df_new_ESP.columns}")

    ### EZS dataset
    if verbose:
        print(" - Get data from EZS...")
    df_new_EZS = pd.read_csv(os.path.join(data_dir, "data_EZS.csv"))
    df_new_EZS = df_new_EZS.rename(columns={
        "MolecularFormula": "molecule",
        "ConnectivitySMILES": "SMILES_isomeric_1"
    })
    if verbose:
        print(f" - EZS dataset size: {len(df_new_EZS)}")
        print(f"    columns: {df_new_EZS.columns}")

    ### Combine datasets
    factor = 3  # augmentation factor
    # count original positive labels
    orig_pos = df_original[df_original["is_active"] == 1]
    n_target = int(factor * len(orig_pos))
    original_pairs = set(zip(df_original["UGT_ID"], df_original["molecule"]))
    # keep only new pairs (not in original)
    df_ESP_tmp = df_new_ESP[~df_new_ESP.apply(
        lambda row: (row["UGT_ID"], row["molecule"]) in original_pairs, axis=1
    )]
    df_EZS_tmp = df_new_EZS[~df_new_EZS.apply(
        lambda row: (row["UGT_ID"], row["molecule"]) in original_pairs, axis=1
    )]
    # take sample from ESP and EZS
    n_each = n_target // 2
    df_ESP = df_ESP_tmp.sample(n=min(n_each, len(df_ESP_tmp)), random_state=42)
    df_EZS = df_EZS_tmp.sample(n=min(n_each, len(df_EZS_tmp)), random_state=42)
    df_all = pd.concat([df_original, df_ESP, df_EZS], ignore_index=True)
    if verbose:
        print(f" - Original positives: {len(orig_pos)}")
        print(f" - took {len(df_ESP)} (pos.) samples from ESP")
        print(f" - took {len(df_EZS)} (pos.) samples from EZS")
        print(f" - Merged dataset shape: {df_all.shape}")

    return df_all

def create_csv(verbose=False):
    from sqlalchemy import create_engine

    # Load environment variables
    load_dotenv()

    # Read values from .env
    db_file = os.getenv("ACCESS_DB_PATH")
    password = os.getenv("ACCESS_DB_PASSWORD")
    location = Path(__file__).parent.parent.parent / "data"

    absolute_path = os.path.abspath(db_file)

    # Check if file exists
    if not os.path.exists(absolute_path):
        raise FileNotFoundError(f" - File not found at: {absolute_path}")

    # Build ODBC connection string
    odbc_conn_str = (
        r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
        f"DBQ={absolute_path};"
        f"PWD={password};"
    )

    # Create SQLAlchemy engine
    engine = create_engine(f"access+pyodbc:///?odbc_connect={odbc_conn_str}")

    # Read UGT table
    df_ugt = pd.read_sql(f"SELECT UGT_ID,UGT_trivial_name, UGT_Nomenclature,nt_seq,prot_seq FROM UGT", engine)
    # Save as CSV
    output_path_ugt = location / "UGT.csv"
    df_ugt.to_csv(output_path_ugt, index=False)
    if verbose:
        print(f" -> Table UGT exported to {output_path_ugt}")

    # Read Activity table
    df_act = pd.read_sql(f"SELECT ID,UGT_trivial_name, substrate,activity FROM Activity WHERE activity <>'missing' ", engine)
    # Save as CSV
    output_path_act = location / "Activity.csv"
    df_act.to_csv(output_path_act, index=False)
    if verbose:
        print(f" -> Table Activity exported to {output_path_act}")

    # Read activity table and get distinct substrates
    df_substrate = pd.read_sql(f"SELECT DISTINCT substrate FROM Activity WHERE activity <>'missing' ", engine)
    if verbose:
        print(" - Fetching PubChem data for substrates...")
    tqdm.pandas()  # enables progress_apply for nice progress bar

    # Apply your function once per substrate and expand dict into columns
    tqdm.pandas()  # enables progress_apply for nice progress bar

    df_substrate[["MolecularFormula", "ConnectivitySMILES"]] = (
        df_substrate["substrate"]
        .progress_apply(lambda name: pd.Series(get_pubchem_info(name)))
    )
    # Save as CSV
    output_path_substrate = location / "Substrate_SMILES.csv"
    df_substrate.to_csv(output_path_substrate, index=False)
    if verbose:
        print(f" -> Substrates exported to {output_path_substrate}")

# Function to query PubChem
def get_pubchem_info(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularFormula,CanonicalSMILES/JSON"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        props = data["PropertyTable"]["Properties"][0]
        return {
            "MolecularFormula": props.get("MolecularFormula"),
            "ConnectivitySMILES": props.get("ConnectivitySMILES")
        }
    except Exception:
        return None

def prepare_kpgt_data(verbose=False):
    """
    Prepare substrate data in KPGT format.
    KPGT expects: data/Substrate/Substrate.csv with 'smiles' column
    """
    # Load substrate SMILES data
    df = pd.read_csv(data_dir / "Substrate_SMILES.csv")

    # KPGT expects a 'smiles' column (lowercase)
    # We'll use the SMILES_isomeric_1 column
    # Note: For feature extraction only, we need a dummy label column
    df_kpgt = pd.DataFrame({
        'smiles': df['SMILES_isomeric_1'].values,
        'dummy_label': 0  # Dummy label for feature extraction
    })

    # Remove rows with missing SMILES
    df_kpgt = df_kpgt.dropna(subset=['smiles'])

    if verbose:
        print(f" - Prepared {len(df_kpgt)} substrates with valid SMILES")
        print(f" - First few rows:")
        print(df_kpgt.head())

    # Save to KPGT format
    # which expects data_path/dataset_name/dataset_name.csv
    # Set up paths
    substrate_dir = data_dir / "Substrate"
    # Create Substrate directory if it doesn't exist
    substrate_dir.mkdir(exist_ok=True)
    output_path = substrate_dir / "Substrate.csv"
    df_kpgt.to_csv(output_path, index=False)
    if verbose:
        print(f" -> Saved to: {output_path}")

def binarize_activity(df, label_col="activity"):
    """Convert multi-level activity values to binary (active/inactive)."""
    active_labels = ["low", "medium", "high", "low, high", "low, medium", "medium, high"]
    df["is_active"] = df[label_col].apply(lambda x: 1 if x in active_labels else 0)
    return df

def create_fasta_file(verbose=False):

    ugt_path = data_dir / "UGT.csv"

    if not ugt_path.exists():
        raise FileNotFoundError(f"UGT CSV not found at: {ugt_path.resolve()}")

    # Read CSV
    df = pd.read_csv(ugt_path)

    # Open a new FASTA file
    output_path = data_dir / "UGT.fasta"

    # Create FASTA
    with output_path.open("w", encoding="utf8") as fasta_file:
        for _, row in df.iterrows():
            # prefer a human-readable name
            header = row.get("UGT_trivial_name")
            # prefer nucleotide sequence
            sequence = row.get("prot_seq")
            if pd.isna(sequence) or sequence is None or header is None:
                # skip rows without sequence or header
                continue
            fasta_file.write(f">{header}\n{sequence}\n")

    if verbose:
        print(f" -> Wrote FASTA to: {output_path}")



