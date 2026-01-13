import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

# matlab dir
matlab_dir = Path(__file__).parent/ "matlab"

# data directory
data_dir = Path(__file__).parent.parent / "data"

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

def write_train_fasta(train_df, out_fasta):
    records = []
    for enzyme, seq in train_df.groupby("UGT_ID")["prot_seq"].first().items():
        # remove all whitespace/newlines
        clean_seq = "".join(seq.split())
        records.append(
            SeqRecord(Seq(clean_seq), id=str(enzyme), description="")
        )
    SeqIO.write(records, out_fasta, "fasta")

def write_query_fasta(enzyme, seq, path):
    with open(path, "w") as f:
        f.write(">" + str(enzyme) + "\n")
        f.write(seq + "\n")

def find_relaxed_core(df, min_substrates=10, min_enzymes=5, max_iter=20):
    core = df.copy()

    for _ in range(max_iter):
        changed = False

        enz_counts = core.groupby("UGT_ID")["substrate"].nunique()
        keep_enz = enz_counts[enz_counts >= min_substrates].index
        if len(keep_enz) < core["UGT_ID"].nunique():
            core = core[core["UGT_ID"].isin(keep_enz)]
            changed = True

        sub_counts = core.groupby("substrate")["UGT_ID"].nunique()
        keep_sub = sub_counts[sub_counts >= min_enzymes].index
        if len(keep_sub) < core["substrate"].nunique():
            core = core[core["substrate"].isin(keep_sub)]
            changed = True

        if not changed:
            break

    return core

def get_substrate_features(substrates_df):
    COOH_SMARTS = Chem.MolFromSmarts("C(=O)[OH]")
    OH_SMARTS = Chem.MolFromSmarts("[OH]")
    AMINE_SMARTS = Chem.MolFromSmarts("[NX3;H2,H1;!$(NC=O)]")

    rows = []

    for _, row in substrates_df.drop_duplicates("substrate").iterrows():
        mol = Chem.MolFromSmiles(row["SMILES_isomeric_1"])
        if mol is None:
            raise ValueError(f"Invalid SMILES: {row['SMILES_isomeric_1']}")

        feats = {
            "substrate": row["substrate"],
            "logP": Crippen.MolLogP(mol),
            "area": rdMolDescriptors.CalcTPSA(mol),
            "vol": Descriptors.MolWt(mol), # proxy
            "cooh": int(mol.HasSubstructMatch(COOH_SMARTS)),
            "numOH": Descriptors.NumHDonors(mol),
        }

        rows.append(feats)

    return pd.DataFrame(rows).set_index("substrate")

def build_interaction_matrix(all_pairs_df, out_file, DT=False, test_frac=0.2, seed=0):
    # get core (interactions between substrates and enzymes that have no missing combinations)
    core_df = find_relaxed_core(all_pairs_df)

    enzymes = core_df["UGT_ID"].unique()
    substrates = sorted(core_df["substrate"].unique())

    print(
        f"Dense core: "
        f"{len(enzymes)} enzymes × {len(substrates)} substrates"
    )

    # split dataframes
    if DT:
        # split substrates (not interactions)
        rng = np.random.default_rng(seed)
        n_test = int(len(substrates) * test_frac)
        test_substrates = set(rng.choice(substrates, size=n_test, replace=False))
        train_substrates = set(substrates) - test_substrates

        core_train_df = core_df[core_df["substrate"].isin(train_substrates)].copy()
        core_test_df = core_df[core_df["substrate"].isin(test_substrates)].copy()

        # build matrix from train only
        X = pd.DataFrame(2, index=sorted(train_substrates), columns=sorted(enzymes))
    else:
        # split enzymes (not interactions)
        rng = np.random.default_rng(seed)
        n_test = int(len(enzymes) * test_frac)
        test_enzymes = set(rng.choice(enzymes, size=n_test, replace=False))
        train_enzymes = set(enzymes) - test_enzymes

        core_train_df = core_df[core_df["UGT_ID"].isin(train_enzymes)].copy()
        core_test_df = core_df[core_df["UGT_ID"].isin(test_enzymes)].copy()

        # build matrix from train only
        X = pd.DataFrame(2, index=substrates, columns=sorted(train_enzymes))

    for _, row in core_train_df.iterrows():
        X.loc[row["substrate"], row["UGT_ID"]] = row["is_active"]

    # minimal required metadata
    meta = pd.DataFrame({
        "Name": substrates
    })
    if DT:
        # fill the substrate features
        feats_df = get_substrate_features(all_pairs_df[['substrate', 'molecule', 'SMILES_isomeric_1']])

        # Merge on substrate name
        meta = meta.merge(feats_df, left_on="Name", right_index=True, how="left")

        # Optional: reorder columns so ID/Name are first
        cols = ["Name"] + [c for c in feats_df.columns]
        meta = meta[cols]

    full = pd.concat([meta, X.reset_index(drop=True)], axis=1)
    full.to_csv(out_file, sep="\t", index=False)

    # return test interactions for evaluation
    return core_test_df, core_train_df, meta


def read_gtpredict_output(path):
    """
    Returns dict: substrate -> predicted_label (1/0)
    """
    preds = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip metadata
            if line.startswith("Using"):
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            label = parts[-1]
            substrate = " ".join(parts[:-1])

            if label == "Yes":
                preds[substrate] = 1
            elif label == "No":
                preds[substrate] = 0

    return preds

def read_gtpredict_dt_output(path):
    """
    Returns dict: enzyme id -> predicted_label (1/0)
    """
    preds = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Skip metadata
            if line.startswith("Substrate"):
                continue

            parts = line.split()

            # Expect: [enzyme_id, Yes|No]
            if len(parts) < 2:
                continue

            enzyme_id = parts[0]
            label = parts[-1]

            if label == "Yes":
                preds[enzyme_id] = 1
            elif label == "No":
                preds[enzyme_id] = 0

    return preds

def compute_accuracy(gt_labels, true_df, column):
    y_true = []
    y_pred = []

    for _, row in true_df.iterrows():
        molecule = str(row[column])
        if molecule in gt_labels:
            y_true.append(row.is_active)
            y_pred.append(gt_labels[molecule])

    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)

def PredictEnzyme(df):
    # prepare data for GTPredict
    test_df, train_df, meta = build_interaction_matrix(all_pairs_df=df, out_file=matlab_dir / "gt_interaction.txt")

    # Check intersection
    train_enzymes = set(train_df["UGT_ID"].unique())
    test_enzymes = set(test_df["UGT_ID"].unique())
    overlap = train_enzymes & test_enzymes

    if overlap:
        print(f"Warning! These test enzymes are also in train: {overlap}")
    else:
        print("All test enzymes are unseen in the training matrix")

    write_train_fasta(train_df, matlab_dir / "gt_train.fasta")

    # get GTPredict prediction on the labels
    matlab_cmd = [
        "/usr/local/MATLAB/R2024b/bin/matlab",
        "-nodisplay",
        "-batch",
        f"addpath('{matlab_dir}'); "
        f"PredictEnzymeInteraction_NN('{matlab_dir}/data/gt_interaction.txt', "
        f"'{matlab_dir}/data/gt_train.fasta', "
        f"'{matlab_dir}/data/query.fasta', "
        f"'{matlab_dir}/data/gtpredict_prediction.csv')"
    ]

    accs = []

    for enzyme_id, df in test_df.groupby("UGT_ID"):
        print(f"Prediction for test enzyme number: {enzyme_id} ({df['UGT_Nomenclature'].iloc[0]})")

        seq = df["prot_seq"].iloc[0]

        # write query FASTA
        write_query_fasta(enzyme_id, seq, matlab_dir / "data" /  "query.fasta")

        # run MATLAB
        subprocess.run(matlab_cmd, check=True)

        # read predictions
        pred_dict = read_gtpredict_output(matlab_dir / "data" / "gtpredict_prediction.csv")

        # evaluate
        accs.append(compute_accuracy(gt_labels=pred_dict, true_df=df, column='substrate'))

    print("GT-Predict accuracy:", sum(accs) / len(accs))

def PredictAcceptor(df):
    # prepare data for GTPredict
    test_df, train_df, meta = build_interaction_matrix(all_pairs_df=df,
                                                 out_file=matlab_dir / "data" / "gt_dt_interaction.txt",
                                                 DT=True)

    # Check intersection
    train_substrates = set(train_df["substrate"].unique())
    test_substrates = set(test_df["substrate"].unique())
    overlap = train_substrates & test_substrates

    if overlap:
        print(f"Warning! These test substrates are also in train: {overlap}")
    else:
        print("All test substrates are unseen in the training matrix")

    accs = []

    for substrate, df in test_df.groupby("substrate"):
        print(f"Prediction for test substrate: {substrate}")

        # Create MATLAB struct literal as string
        query = meta[meta["Name"] == substrate].iloc[0]

        # get GTPredict prediction on the labels
        matlab_cmd = [
            "/usr/local/MATLAB/R2024b/bin/matlab",
            "-nodisplay",
            "-batch",
            f"addpath('{matlab_dir}');"
            f"query.name = '{substrate}'; "
            f"query.logP = {query['logP']}; "
            f"query.area = {query['area']}; "
            f"query.vol = {query['vol']}; "
            f"query.cooh = {query['cooh']}; "
            f"query.numOH = {query['numOH']}; "
            f"PredictAcceptorInteraction_DT('{matlab_dir}/data/gt_dt_interaction.txt', "
            f"query, "
            f"'{matlab_dir}/data/gtpredict_dt_prediction.csv')"
        ]

        # run MATLAB
        subprocess.run(matlab_cmd, check=True)

        # read predictions
        pred_dict = read_gtpredict_dt_output(matlab_dir / "data" / "gtpredict_dt_prediction.csv")

        # evaluate
        accs.append(compute_accuracy(gt_labels=pred_dict, true_df=df, column='UGT_ID'))

    print("GT-Predict accuracy:", sum(accs) / len(accs))

def main():
    # get test splits
    splits = pd.read_csv(f"{data_dir}/split.csv")

    PredictEnzyme(splits)

    PredictAcceptor(splits)

    # TODO: get our prediction and compare


if __name__ == "__main__":
    main()