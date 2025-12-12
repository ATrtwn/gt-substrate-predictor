import os
from pathlib import Path
from src.data.preprocessing import binarize_activity
import pandas as pd
import requests
from Bio import Entrez, SeqIO
from multiprocessing import Pool

Entrez.email = "your.email@example.com"

# data directory
data_dir = Path(__file__).parent.parent / "data"

def get_refseq_mrna_accessions(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    r = requests.get(url)

    if r.status_code != 200:
        return []

    data = r.json()

    refseq_ids = []

    for xref in data.get("uniProtKBCrossReferences", []):
        db = xref.get("database")
        props = {p.get("key"): p.get("value") for p in xref.get("properties", [])}

        if db == "EMBL" and props.get("MoleculeType") == "mRNA":
            refseq_ids.append(xref["id"].split('.')[0])  # remove version
        elif db == "RefSeq" and "NucleotideSequenceId" in props:
            acc = props["NucleotideSequenceId"].split('.')[0]
            if acc.startswith(("NM_", "XM_")):
                refseq_ids.append(acc)

    return list(set(refseq_ids))

def fetch_sequence_from_ncbi(acc):
    """
    Returns nucleotide sequence string or None.
    Rejects genomic sequences >10,000 bp.
    """
    try:
        # try mRNA FASTA
        handle = Entrez.efetch(db="nucleotide", id=acc, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        seq = str(record.seq)
        handle.close()

        if len(seq) > 10000:
            return None  # probably genomic, reject

        return seq
    except:
        return None

def replace_nt_sequences(df, out_path):

    new_nt_seqs = []

    uniprot_col = df["uniprot"].to_numpy()  # very fast!
    n = len(df)

    for idx in range(n):
        if idx % 100 == 0:
            print(f"at row {idx}/{n}")

        uniprot = uniprot_col[idx]

        accs = get_refseq_mrna_accessions(uniprot)

        if not accs:
            new_nt_seqs.append(None)
            continue

        seq = None
        for acc in accs:
            seq = fetch_sequence_from_ncbi(acc)
            if seq is not None:
                break

        new_nt_seqs.append(seq)

    df["nt_seq"] = new_nt_seqs

    df.to_csv(out_path, index=False)
    print(f"Saved updated dataset to: {out_path}")

    return df


def split_into_batches(df, n_batches=10):
    batch_size = (len(df) + n_batches - 1) // n_batches
    return [
        df.iloc[i * batch_size:(i + 1) * batch_size].copy()
        for i in range(n_batches)
    ]


def process_batch(args):
    batch_df, batch_idx, out_dir = args
    out_file = os.path.join(out_dir, f"df_EZS_fixed_batch{batch_idx}.csv")

    # --- SKIP IF FILE ALREADY EXISTS AND MATCHES ROW COUNT ---
    if os.path.exists(out_file):
        try:
            existing = pd.read_csv(out_file)

            # Check row count
            if len(existing) == len(batch_df):
                print(f"Batch {batch_idx}: already processed, skipping.")
                return out_file
            else:
                print(f"Batch {batch_idx}: file exists but row count mismatch "
                      f"({len(existing)} vs {len(batch_df)}), reprocessing...")
        except Exception as e:
            print(f"Batch {batch_idx}: error reading existing file, reprocessing... {e}")

    # ------------------
    processed = replace_nt_sequences(batch_df, out_file)
    processed.to_csv(out_file, index=False)
    return out_file


def process_in_parallel(df, out_dir, n_batches=10):
    # Split into batches
    batches = split_into_batches(df, n_batches)

    # Prepare argument tuples
    tasks = [(batches[i], i, out_dir) for i in range(len(batches))]

    print(f"Launching {len(batches)} parallel workers...")

    # Run in parallel
    with Pool(processes=n_batches) as pool:
        results = pool.map(process_batch, tasks)

    print("Finished. Output files:")
    for r in results:
        print("  ", r)

    return results

def main(dataset=''):
    df_original = pd.read_csv(os.path.join(data_dir, "merged.csv"))
    # binarize
    df_original = binarize_activity(df_original)
    df_original = df_original.sort_values('is_active', ascending=False)
    df_original = df_original.drop_duplicates(subset=['UGT_ID', 'substrate'], keep='first')
    df_original = df_original[['UGT_ID', 'substrate', 'UGT_Nomenclature',
                               'nt_seq', 'prot_seq', 'MolecularFormula', 'ConnectivitySMILES',
                               'is_active']].drop_duplicates()
    df_original['dataset'] = 'original'
    print(f"original dataset size: {len(df_original)}")
    print(df_original.columns)

    if dataset=='ESP':
        ### ESP dataset
        print("Get data from ESP...")
        df_new_ESP = pd.read_pickle(os.path.join(data_dir, "df_final_ESP.pkl"))
        nt_seq_df_ESP = pd.read_pickle(os.path.join(data_dir, "nt_seq_ESP.pkl"))
        # Merge the unique nt_seq values
        df_new_ESP = df_new_ESP.merge(
            nt_seq_df_ESP,
            left_on="nt_seq_id",
            right_on="id",
            how="left"
        )
        df_new_ESP['uniprot'] = df_new_ESP['UGT_ID']

        print("Starting parallel nt-seq fetching...")

        batch_files = process_in_parallel(
            df_new_ESP,
            out_dir=data_dir,
            n_batches=10
        )

        # assume batch_files is a list of file paths returned from process_in_parallel
        all_batches = []

        for f in batch_files:
            df_batch = pd.read_csv(f)  # or pd.read_pickle if you saved as pickle
            all_batches.append(df_batch)

        # Concatenate into one DataFrame
        df_merged = pd.concat(all_batches, ignore_index=False)

        # Check how many rows have no nt_seq
        missing_nt_seq = df_merged['nt_seq'].isna().sum()
        total_rows = len(df_merged)

        print(f"Total rows: {total_rows}")
        print(f"Rows missing nt_seq: {missing_nt_seq}")
        print(f"Fraction missing: {missing_nt_seq / total_rows:.2%}")

        df_merged = df_merged.dropna(subset=['nt_seq'])
        print(f"ESP dataset size: {len(df_merged)}")
        print(df_merged.columns)
        save_path = os.path.join(data_dir, "df_ESP.csv")
        df_merged.to_csv(save_path, index=False)

    elif dataset == 'EZS':
        ### EZS dataset
        print("Get data from EZS...")
        df_new_EZS = pd.read_pickle(os.path.join(data_dir, "df_final_EZS.pkl"))
        nt_seq_df_EZS = pd.read_pickle(os.path.join(data_dir, "nt_seq_EZS.pkl"))
        # Merge the unique nt_seq values
        df_new_EZS = df_new_EZS.merge(
            nt_seq_df_EZS,
            left_on="nt_seq_id",
            right_on="id",
            how="left"
        )
        df_new_EZS['uniprot'] = df_new_EZS['UGT_ID']

        print("Starting parallel nt-seq fetching...")

        batch_files = process_in_parallel(
            df_new_EZS,
            out_dir=data_dir,
            n_batches=10
        )

        # assume batch_files is a list of file paths returned from process_in_parallel
        all_batches = []

        for f in batch_files:
            df_batch = pd.read_csv(f)  # or pd.read_pickle if you saved as pickle
            all_batches.append(df_batch)

        # Concatenate into one DataFrame
        df_merged = pd.concat(all_batches, ignore_index=False)

        # Check how many rows have no nt_seq
        missing_nt_seq = df_merged['nt_seq'].isna().sum()
        total_rows = len(df_merged)

        print(f"Total rows: {total_rows}")
        print(f"Rows missing nt_seq: {missing_nt_seq}")
        print(f"Fraction missing: {missing_nt_seq / total_rows:.2%}")

        df_merged = df_merged.dropna(subset=['nt_seq'])
        print(f"EZS dataset size: {len(df_merged)}")
        print(df_merged.columns)
        save_path = os.path.join(data_dir, "df_EZS.csv")
        df_merged.to_csv(save_path, index=False)
    else:
        pass


if __name__ == "__main__":
    main(dataset='EZS')