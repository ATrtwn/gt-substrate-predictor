import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from src.prior_work.ESP.code.ES_prediction import ESP_predicton

def evaluate_esp(
        split_name: str, 
        df: pd.DataFrame, 
        prot_seq_col: str = "prot_seq", 
        substrate_col: str = "ConnectivitySMILES", 
        is_active_col: str = 'is_active'
    ):
    df = df[:3]
    # Ground truth
    y_true = df[is_active_col].values

    # Run ESP
    esp_out = ESP_predicton(
        enzyme_list=df[prot_seq_col].tolist(),
        substrate_list=df[substrate_col].tolist()
    )

    print(esp_out)

def main():
    data_dir = Path(__file__).parent.parent / "data"

    c1_val = pd.read_csv(data_dir / "C1_val.csv")
    c2_val = pd.read_csv(data_dir / "C2_val.csv")
    c3_val = pd.read_csv(data_dir / "C3_val.csv")

    c1_test = pd.read_csv(data_dir / "C1_test.csv")
    c2_test = pd.read_csv(data_dir / "C2_test.csv")
    c3_test = pd.read_csv(data_dir / "C3_test.csv")

    evaluate_esp("C1 Val", c1_val)
    evaluate_esp("C2 Val", c2_val)
    evaluate_esp("C3 Val", c3_val)
    evaluate_esp("C1 Test", c1_test)
    evaluate_esp("C2 Test", c2_test)
    evaluate_esp("C3 Test", c3_test)

if __name__ == "__main__":
    main()