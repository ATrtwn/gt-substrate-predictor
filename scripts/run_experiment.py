import logging
import warnings
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional

from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parent.parent))
import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

from src.training.evaluation import compute_f1_score

import numpy as np
import pandas as pd
import optuna
import json
import joblib

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Add project root (two folders up from scripts/) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_split import stratified_split_by_entities, check_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

from src.utils.helper_function import get_params, setup_logging, nano_id, sample_params
from src.training.train import SklearnTrainer

from datetime import datetime

# data directory
ROOT = Path(__file__).parent.parent


MODEL_MAPPING = {
    "majority_classifier": DummyClassifier,
    "random_classifier": DummyClassifier,
    "ridge_classifier": RidgeClassifier,
    "decision_tree" : DecisionTreeClassifier,
    "logistic_regression" : LogisticRegression,
    "random_forest" : RandomForestClassifier,
    "svm" : SVC,
    "gradient_boosting" : GradientBoostingClassifier,
    "k_nearest_neighbors" : KNeighborsClassifier
    # add more models here
}

def run_sklearn_experiment(
    model_name: str,
    model_params: dict[str, Any],
    substrate_name:str,
    protein_name:str,
    wandb_mode: str = "offline",
    project: str = "gt-substrate-predictor",
    sweep: bool = False,
    concatenation_path:str=None,
) -> None:
    

    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(MODEL_MAPPING.keys())}")

    sk_model = MODEL_MAPPING[model_name](**model_params)

    # Wrap in SklearnTrainer
    trainer = SklearnTrainer(model=sk_model)

    # --- Load the embeddings here ---
    # Load data
    data_dir = Path(__file__).parent.parent
    concatenated_embeddings = np.load(data_dir / concatenation_path / f'X_{substrate_name}.npy')
    activity = np.load(data_dir / concatenation_path / f'y_{substrate_name}.npy')
    meta_name = f"metadata_{substrate_name}.csv"
    metadata = pd.read_csv(data_dir / concatenation_path / meta_name)
    metadata = metadata.rename(columns={'activity': 'is_active'})
    df = pd.read_csv(f"{ROOT}/data/split.csv")
    
    # Convert to binary: 0 if "none", else 1
    # Auto-detect binarization
    if activity.dtype.kind in {'U', 'S', 'O'}:
        # String or object: treat 'none' as negative, else positive
        activity_binary = (activity != "none").astype(int)
    else:
        # Numeric: assume already binarized (0/1)
        activity_binary = activity.astype(int)
    logging.info(f"Loaded {len(activity_binary)} samples")
    logging.info(f"Embedding dimension: {concatenated_embeddings.shape[1]}")
    logging.info(f"Class distribution: {np.bincount(activity_binary)}")
    def get_embeddings_for_split(split_df, metadata, concatenated_embeddings):
        split_cols = {col.lower(): col for col in split_df.columns}
        meta_cols = {col.lower(): col for col in metadata.columns}
        merge_cols = []
        if 'ugt_id' in meta_cols and ('ugt_id' in split_cols or 'ugt_id' in [c.lower() for c in split_df.columns]):
            merge_cols.append(('UGT_ID' if 'UGT_ID' in split_df.columns else split_cols.get('ugt_id', 'ugt_id'), meta_cols['ugt_id']))
        elif 'ugt_id' in meta_cols and 'UGT_ID' in split_cols:
            merge_cols.append(('UGT_ID', meta_cols['ugt_id']))
        if 'substrate' in meta_cols and 'substrate' in split_cols:
            merge_cols.append(('substrate', 'substrate'))
        if merge_cols:
            left_on = [mc[0] for mc in merge_cols]
            right_on = [mc[1] for mc in merge_cols]
            merged = pd.merge(split_df, metadata.reset_index(), left_on=left_on, right_on=right_on, how='inner')
            indices = merged['index'].values.astype(int)
        else:
            indices = metadata.index.isin(split_df.index).nonzero()[0]
        return concatenated_embeddings[indices], activity_binary[indices]
    
    
    logging.info("Creating data splits...")
    logging.info("Load Split ...")
    mapping = df[['UGT_ID', 'substrate', 'cluster_id','dataset']].drop_duplicates()

    # 2. Merge into metadata
    metadata = metadata.merge(
        mapping,
        left_on=['ugt_id', 'substrate'],
        right_on=['UGT_ID', 'substrate'],
        how='left'
    )
    metadata = metadata.drop(columns=['ugt_id'])

    splits = pd.read_csv(f"{ROOT}/data/split.csv")
    train = splits[splits["split"] == "train"]
    val1 = splits[splits["split"]=="C1_val"]
    val2 = splits[splits["split"]=="C2_val"]
    val3 = splits[splits["split"]=="C3_val"]
    val = pd.concat([val1, val2, val3], axis=0)

    c1 = splits[splits["split"] == "C1_test"]
    c2 = splits[splits["split"] == "C2_test"]
    c3 = splits[splits["split"] == "C3_test"]
    test_sets = {"C1_test": c1, "C2_test": c2, "C3_test": c3}
  

     # Get embeddings for each split
    metadata=metadata[0:len(concatenated_embeddings)]
    train_emb, train_labels = get_embeddings_for_split(train, metadata, concatenated_embeddings)
    val_emb, val_labels = get_embeddings_for_split(val, metadata, concatenated_embeddings)
    c1_emb, c1_labels = get_embeddings_for_split(c1, metadata, concatenated_embeddings) if c1 is not None else (None, None)
    c2_emb, c2_labels = get_embeddings_for_split(c2, metadata, concatenated_embeddings) if c2 is not None else (None, None)
    c3_emb, c3_labels = get_embeddings_for_split(c3, metadata, concatenated_embeddings) if c3 is not None else (None, None)
 



    # Normalize embeddings - fit on train, transform all
    logging.info("Normalizing embeddings with StandardScaler...")
    scaler = StandardScaler()
    train_emb = scaler.fit_transform(train_emb)
    val_emb = scaler.transform(val_emb)
    c1_emb = scaler.transform(c1_emb)
    c2_emb = scaler.transform(c2_emb)
    c3_emb = scaler.transform(c3_emb)
    

   
    table = wandb.Table(columns=["Subset", "Class", "Frequency"])

    for name, subset in [("Training", train), ("val", val), ("C1", c1), ("C2", c2), ("C3", c3)]:
        counts = subset["is_active"].value_counts(normalize=True).sort_index()
        for cls, freq in counts.items():
            table.add_data(name, cls, freq)

    
    test_dict_sets = {
        "C1_test": [c1_emb,c1_labels],
        "C2_test": [c2_emb,c2_labels],
        "C3_test": [c3_emb,c3_labels]
    }

    


    # Fit the model
    logging.info("Start training" + model_name + " with params :" + str(model_params))
   
    history = trainer.fit(train_emb, train_labels, val_emb, val_labels)

    # --- 2. SAVE MODEL LOCALLY ---
    local_dir = ROOT / "reports" / "results_sklearn" / f"{model_name}_substrate-{substrate_name}_protein-{protein_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    local_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(sk_model, local_dir / "model.joblib")
    joblib.dump(scaler, local_dir / "scaler.joblib")

    metrics_list = []
    pred_data = []
    return_f1 = 0.0

    for set_name, test_set in test_dict_sets.items():
        emb,indices = test_set
        y_true = activity_binary[indices]
        y_prob = trainer.predict(emb)
        y_pred = (y_prob > 0.5).astype(int)

        # Calculate Metrics
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        
        if set_name == "val": return_f1 = f1

        metrics_list.append({"Split": set_name, "Accuracy": acc, "F1": f1, "MCC": mcc})
        
        # Collect predictions
        for i, idx in enumerate(indices):
            pred_data.append([set_name, idx, y_true[i], y_prob[i], y_pred[i]])

    # --- 3. SAVE DATA TABLES LOCALLY ---
    metrics_df = pd.DataFrame(metrics_list)
    preds_df = pd.DataFrame(pred_data, columns=["Split", "Index", "True", "Prob", "Pred"])
    
    metrics_df.to_csv(local_dir / "metrics.csv", index=False)
    preds_df.to_csv(local_dir / "predictions.csv", index=False)
    
    with open(local_dir / "config.json", "w") as f:
        json.dump(model_params, f, indent=4)

    # Log to WandB
    # wandb.log({
    #     "metrics_table": wandb.Table(dataframe=metrics_df),
    #     "predictions_table": wandb.Table(dataframe=preds_df),
    #     "train_loss": history["train_loss"][-1]
    # })

    #run.finish()
    return return_f1




# def train_and_log(params: dict[str, Any] = None) -> None:
#     logging.info("Starting experiment...")
#     model_classes = params["models"]
#     tasks = []
#     for model_name, model_params in model_classes.items():
#         logging.info(f"Preparing model: {model_name} with params: {model_params}")
#         model_params_copy = model_params.copy()

#         tasks.append(
#             delayed(run_sklearn_experiment)(
#                 model_name=model_name,
#                 model_params=model_params_copy,
#                 substrate_name=params["substrate_name"],
#                 protein_name=params["protein_name"],
#                 project=params["project"],
#                 wandb_mode=params["wandb_mode"],
#                 concatenation_path=params.get("concatenation_path", None)
#             )
#         )
#         try:
#             with parallel_config(
#                 backend="loky",
#                 n_jobs=params["n_jobs"],
#                 max_nbytes=params["max_mem"],
#             ):
#                 Parallel(n_jobs=params["n_jobs"], timeout=None)(tasks)
#         except Exception:
#             logging.error("Error occurred, attempting to clean up...")
#             raise
        

def train_and_log(params: dict[str, Any] = None) -> None:
    logging.info("Starting experiment in SERIAL mode for debugging...")
    
    model_classes = params["models"]
    
    # Iterate through models one by one
    for model_name, model_params in model_classes.items():
        logging.info(f"--- Running Model: {model_name} ---")
        
        # We still copy to be safe, though less critical in a single thread
        model_params_copy = model_params.copy()

        try:
            # Call the experiment function directly instead of using delayed()
            run_sklearn_experiment(
                model_name=model_name,
                model_params=model_params_copy,
                substrate_name=params["substrate_name"],
                protein_name=params["protein_name"],
                project=params["project"],
                wandb_mode=params["wandb_mode"],
                concatenation_path=params.get("concatenation_path", None)
            )
            logging.info(f"Successfully finished {model_name}")
            
        except Exception as e:
            # This will now show you the full, useful traceback
            logging.error(f"Failed during model: {model_name}")
            logging.exception(e) 
            raise  # Stop execution immediately so you can inspect the error
def main():
    train_and_log(get_params("scikit_learn"))


if __name__ == "__main__":
    setup_logging()
    main()
