import pandas as pd

from pathlib import Path
import subprocess

# matlab dir
matlab_dir = Path(__file__).parent.parent/ "src" / "matlab"

# data directory
data_dir = Path(__file__).parent.parent / "data"

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

from src.matlab.prepare_data import build_interaction_matrix_enzyme, build_interaction_matrix_acceptor

def write_train_fasta(train_df, out_fasta):
    records = []
    for enzyme, seq in train_df.groupby("UGT_ID")["prot_seq"].first().items():
        # remove all whitespace/newlines
        clean_seq = "".join(seq.split())
        id_str = str(enzyme)
        records.append(
            SeqRecord(Seq(clean_seq), id=id_str, description="")
        )
    SeqIO.write(records, out_fasta, "fasta")

def write_query_fasta(enzyme, seq, path):
    #with open(path, "w") as f:
    #    id_str = ">" + str(enzyme) + "\n"
    #    f.write(id_str)
    #    f.write(seq + "\n")

    # remove all whitespace/newlines
    clean_seq = "".join(seq.split())
    id_str = str(enzyme)
    records = []
    records.append(
        SeqRecord(Seq(clean_seq), id=id_str, description="")
    )
    SeqIO.write(records, path, "fasta")

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
            elif label == "Missing":
                preds[substrate] = -1

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
    """
    Compute accuracy for a given molecule (enzyme or substrate) over all its partners.
    Ignores missing predictions (marked as -1).
    Returns: accuracy (float), number of missing predictions (int)
    """
    print(f"  Compute accuracy for single enzyme/substrate query")

    y_true, y_pred = [], []
    skip_cnt = 0

    for _, row in true_df.iterrows():
        partner = str(row[column])
        if partner in gt_labels:
            pred = gt_labels[partner]
            truth = row.is_active
            if pred == -1:
                skip_cnt += 1
                continue
            y_true.append(truth)
            y_pred.append(pred)
        else:
            print(f"  No prediction available for {partner}")
            skip_cnt += 1

    if y_true:
        acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    else:
        acc = None  # No valid predictions at all

    print(f"  Skipped {skip_cnt} rows with no prediction")
    return acc, skip_cnt

def build_groups_raw(row):
    fam = int(row["Family"])

    if fam == 1:  # Flavonoids
        return [
            row["F3-OH"],
            row["F5-OH"],
            row["F6-OH"],
            row["F7-OH"],
            row["F13-OH"],
            row["F14-OH"],
        ]

    elif fam == 2:  # Coumarins
        return [
            row["Cm6-OH"],
            row["Cm7-OH"],
        ]

    elif fam == 3:  # Cytokinins
        return [
            row["Ck3-N"],
            row["Ck7-N"],
            row["Ck-OH"],
        ]

    elif fam == 4:  # Cinnamic acids
        return [
            row["Cn2-OH"],
            row["Cn3-OH"],
            row["Cn4-OH"],
        ]

    else:
        return []  # families with no scaffold features

def PredictEnzyme(df):
    print(f"\n +++ Prediction for GTPredict performance on Enzyme queries +++ ")

    # prepare data for GTPredict
    test_df, train_df, meta = build_interaction_matrix_enzyme(all_pairs_df=df,
                                                              out_file=matlab_dir / "data" / "gt_interaction.txt")

    # Check intersection
    train_enzymes = set(train_df["UGT_ID"].unique())
    test_enzymes = set(test_df["UGT_ID"].unique())
    overlap = train_enzymes & test_enzymes

    if overlap:
        print(f"Warning! These test enzymes are also in train: {overlap}")

    total_rows = len(train_df) + len(test_df)
    print("GT-Predict evaluation summary")
    print("--------------------------------")
    print(
        f"Training rows (interactions): "
        f"{len(train_df)}/{total_rows} "
        f"({len(train_df) / total_rows:.1%})"
    )
    print(
        f"Test rows (interactions):     "
        f"{len(test_df)}/{total_rows} "
        f"({len(test_df) / total_rows:.1%})"
    )

    write_train_fasta(train_df, matlab_dir / "data" / "gt_train.fasta")

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

    set_accs = {}
    set_missing = {}
    set_total_acc = {}

    for test_set in ["C1_test", "C2_test", "C3_test"]:
        set_df = test_df[test_df["split"] == test_set].copy()
        if set_df.empty:
            print(f"No substrates in {test_set}, skipping...")
            continue

        accs = []
        missing_counts = []

        # For total accuracy
        total_correct = 0
        total_checked = 0
        total_missing = 0

        print(f"\n+++ Predictions for {test_set} ({len(set_df)} interactions) +++")

        for enzyme_id, enzyme_df in set_df.groupby("UGT_ID"):
            print(f"Prediction for test enzyme id: {enzyme_id} ({enzyme_df['UGT_Nomenclature'].iloc[0]})")
            print(f"Enzyme has {len(enzyme_df)} interactions in the test data")

            seq = enzyme_df["prot_seq"].iloc[0]

            # write query FASTA
            write_query_fasta(enzyme_id, seq, matlab_dir / "data" /  "query.fasta")

            # run MATLAB
            subprocess.run(matlab_cmd, check=True)

            # read predictions
            pred_dict = read_gtpredict_output(matlab_dir / "data" / "gtpredict_prediction.csv")
            total_preds = len(pred_dict)
            missing_overall = sum([pred_dict[pred] == -1 for pred in pred_dict])
            print(f"Missing predictions in list of all substrates: {missing_overall} / {total_preds}")

            # compute accuracy only on valid predictions
            acc, missing = compute_accuracy(gt_labels=pred_dict, true_df=enzyme_df, column='substrate')

            if acc is not None:
                print(f"Result: {acc:.2f}")
                accs.append(acc)
                n_checked = len(enzyme_df) - missing
                n_correct = int(acc * n_checked)
                total_correct += n_correct
                total_checked += n_checked
            else:
                print("No valid predictions on test set to compute accuracy.")

            missing_counts.append(missing)
            total_missing += missing

        # Store per set
        mean_acc = sum(accs) / len(accs) if accs else 0
        set_accs[test_set] = mean_acc
        set_missing[test_set] = total_missing
        total_acc = total_correct / total_checked if total_checked else 0
        set_total_acc[test_set] = (total_acc, total_correct, total_checked)

    # Summary
    print("\n=== Summary per test set ===")
    for test_set in ["C1_test", "C2_test", "C3_test"]:
        if test_set in set_accs:
            n_interactions = len(test_df[test_df["split"] == test_set])
            mean_acc = set_accs[test_set]
            missing = set_missing[test_set]
            total_acc, n_correct, n_checked = set_total_acc[test_set]
            print(f"{test_set}: mean accuracy = {mean_acc:.2f}, "
                  f"total accuracy = {total_acc:.2f} ({n_correct}/{n_checked}), "
                  f"missing = {missing} / {n_interactions}")


def PredictAcceptor(df):
    print(f"\n +++ Prediction for GTPredict performance on Substrate queries +++ ")

    # prepare data for GTPredict
    test_df, train_df, meta = build_interaction_matrix_acceptor(all_pairs_df=df,
                                                                out_file=matlab_dir / "data" / "gt_dt_interaction.txt")

    # Check intersection
    train_substrates = set(train_df["substrate"].unique())
    test_substrates = set(test_df["substrate"].unique())
    overlap = train_substrates & test_substrates

    if overlap:
        print(f"Warning! These test substrates are also in train: {overlap}")

    total_rows = len(train_df) + len(test_df)
    print("GT-Predict evaluation summary")
    print("--------------------------------")
    print(
        f"Training rows (interactions): "
        f"{len(train_df)}/{total_rows} "
        f"({len(train_df) / total_rows:.1%})"
    )
    print(
        f"Test rows (interactions):     "
        f"{len(test_df)}/{total_rows} "
        f"({len(test_df) / total_rows:.1%})"
    )

    set_accs = {}
    set_missing = {}
    set_total_acc = {}

    for test_set in ["C1_test", "C2_test", "C3_test"]:
        set_df = test_df[test_df["split"] == test_set].copy()
        if set_df.empty:
            print(f"No substrates in {test_set}, skipping...")
            continue

        accs = []
        missing_counts = []

        # For total accuracy
        total_correct = 0
        total_checked = 0
        total_missing = 0

        print(f"\n+++ Predictions for {test_set} ({len(set_df)} interactions) +++")

        for substrate, substrate_df in set_df.groupby("substrate"):
            print(f"Prediction for test substrate: {substrate}")
            print(f"Substrate has {len(substrate_df)} interactions in the test data")

            # Create MATLAB struct literal as string
            query = meta[meta["Name"] == substrate].iloc[0]

            groups_raw = build_groups_raw(query)
            groups_str = " ".join(str(int(x)) for x in groups_raw)

            # get GTPredict prediction on the labels
            matlab_cmd = [
                "/usr/local/MATLAB/R2024b/bin/matlab",
                "-nodisplay",
                "-batch",
                f"addpath('{matlab_dir}');"
                f"query.name = '{substrate}'; "
                f"query.family = {query['Family']}; "
                f"query.logP = {query['LogP']}; "
                f"query.area = {query['AccessibleArea']}; "
                f"query.vol = {query['Volume']}; "
                f"query.cooh = {query['COOH']}; "
                f"query.numOH = {query['Num_OH']}; "
                f"query.groups_raw = [{groups_str}]; "
                f"PredictAcceptorInteraction_DT('{matlab_dir}/data/gt_dt_interaction.txt', "
                f"query, "
                f"'{matlab_dir}/data/gtpredict_dt_prediction.csv')"
            ]

            # run MATLAB
            subprocess.run(matlab_cmd, check=True)

            # read predictions
            pred_dict = read_gtpredict_dt_output(matlab_dir / "data" / "gtpredict_dt_prediction.csv")

            # compute accuracy only on valid predictions
            acc, missing = compute_accuracy(gt_labels=pred_dict, true_df=substrate_df, column='UGT_ID')

            if acc is not None:
                print(f"Result: {acc:.2f}")
                accs.append(acc)
                n_checked = len(substrate_df) - missing
                n_correct = int(acc * n_checked)
                total_correct += n_correct
                total_checked += n_checked
            else:
                print("No valid predictions on test set to compute accuracy.")

            missing_counts.append(missing)
            total_missing += missing

        # Store per set
        mean_acc = sum(accs) / len(accs) if accs else 0
        set_accs[test_set] = mean_acc
        set_missing[test_set] = total_missing
        total_acc = total_correct / total_checked if total_checked else 0
        set_total_acc[test_set] = (total_acc, total_correct, total_checked)

    # Summary
    print("\n=== Summary per test set ===")
    for test_set in ["C1_test", "C2_test", "C3_test"]:
        if test_set in set_accs:
            n_interactions = len(test_df[test_df["split"] == test_set])
            mean_acc = set_accs[test_set]
            missing = set_missing[test_set]
            total_acc, n_correct, n_checked = set_total_acc[test_set]
            print(f"{test_set}: mean accuracy = {mean_acc:.2f}, "
                  f"total accuracy = {total_acc:.2f} ({n_correct}/{n_checked}), "
                  f"missing = {missing} / {n_interactions}")

def main():
    # get test splits
    splits = pd.read_csv(f"{data_dir}/split.csv")
    splits["substrate"] = (
        splits["substrate"]
        .astype(str)
        .str.replace("'", "", regex=False)  # remove apostrophes
        .str.strip()
    )

    PredictEnzyme(splits)

    PredictAcceptor(splits)

    # TODO: get our prediction and compare


if __name__ == "__main__":
    main()