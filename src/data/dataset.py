import pandas as pd
from dotenv import load_dotenv
import os
import requests
from tqdm import tqdm  # progress bar
from pathlib import Path

# Function to query PubChem
def get_pubchem_info(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/MolecularFormula,CanonicalSMILES/JSON"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        props = data["PropertyTable"]["Properties"][0]["MolecularFormula"]
        return props
    except Exception:
        return None

def create_csv():
    from sqlalchemy import create_engine

    # Load environment variables
    load_dotenv()

    # Read values from .env
    db_file = os.getenv("ACCESS_DB_PATH")
    password = os.getenv("ACCESS_DB_PASSWORD")
    location = Path(__file__).parent.parent.parent / "data"

    # Build ODBC connection string
    odbc_conn_str = (
        r"Driver={Microsoft Access Driver (*.mdb, *.accdb)};"
        f"DBQ={db_file};"
        f"PWD={password};"
    )

    # Create SQLAlchemy engine
    engine = create_engine(f"access+pyodbc:///?odbc_connect={odbc_conn_str}")

    # Read UGT table
    df_ugt = pd.read_sql(f"SELECT UGT_ID,UGT_trivial_name, UGT_Nomenclature,nt_seq,prot_seq FROM UGT", engine)
    # Save as CSV
    output_path_ugt = location / "UGT.csv"
    df_ugt.to_csv(output_path_ugt, index=False)
    print(f"✅ Table UGT exported to {output_path_ugt}")

    # Read Activity table
    df_act = pd.read_sql(f"SELECT ID,UGT_trivial_name, substrate,activity FROM Activity WHERE activity <>'None' ", engine)
    # Save as CSV
    output_path_act = location / "Activity.csv"
    df_act.to_csv(output_path_act, index=False)
    print(f"✅ Table Activity exported to {output_path_act}")

    # Read activity table and get distinct substrates
    df_substrate = pd.read_sql(f"SELECT DISTINCT substrate FROM Activity WHERE activity <>'None' ", engine)
    print("Fetching PubChem data for substrates...")
    df_substrate["molecule"] = [get_pubchem_info(name) for name in tqdm(df_substrate["substrate"])]
    # Save as CSV
    output_path_substrate = location / "Substrate.csv"
    df_substrate.to_csv(output_path_substrate, index=False)
    print(f"✅ Substrates exported to {output_path_substrate}")
