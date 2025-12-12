import logging
import warnings
import sys
from functools import partial
from pathlib import Path
from typing import Any, Optional

from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

import wandb
from optuna.integration.wandb import WeightsAndBiasesCallback

wandb.init(project="gt-substrate-predictor", mode="offline")

import numpy as np
import pandas as pd
import optuna

from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# Add project root (two folders up from scripts/) to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_split import stratified_split_by_entities, check_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

from src.utils.helper_function import get_params, setup_logging, nano_id, sample_params
from src.training.train import SklearnTrainer

from datetime import datetime

data_dir = Path(__file__).parent.parent 

MODEL_MAPPING = {
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
    run = (
        wandb.init(
            project=project,
            name=f"{model_name}_substrate-{substrate_name}_protein-{protein_name}_id-{nano_id()}",
            config={
                "model_name": model_name,
                "model_params": model_params,
                "dataset_version": "1.0"
            },
            mode=wandb_mode,
        )
        if not sweep
        else wandb.init(mode=wandb_mode)
    )

    if model_name not in MODEL_MAPPING:
        raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(MODEL_MAPPING.keys())}")

    sk_model = MODEL_MAPPING[model_name](**model_params)

    # Wrap in SklearnTrainer
    trainer = SklearnTrainer(model=sk_model)

    # --- Load the embeddings here ---
    if concatenation_path is not None:
        concatenated_embeddings = np.load( data_dir/concatenation_path / f'X_{substrate_name}.npy')
        activity = np.load(data_dir / concatenation_path / f'y_{substrate_name}.npy')
    meta_name = "metadata_"+substrate_name+"_"+protein_name+".csv"
    metadata = pd.read_csv(data_dir / concatenation_path /  meta_name )
    # Convert activity to binary: 0 if value is the string "None", else 1

    activity = (activity != "none").astype(int)

    protein_col = "UGT_trivial_name"
    substrate_col = "substrate"
    label_col = "activity"

    unique_proteins = metadata[protein_col].unique()
    unique_substrates = metadata[substrate_col].unique()

    wandb.log({
        "unique_proteins": len(unique_proteins),
        "unique_substrates": len(unique_substrates)
    })
    logging.info("Create Split ...")

    splits = stratified_split_by_entities(metadata,
                                          protein_col=protein_col,
                                          substrate_col=substrate_col,
                                          label_col=label_col,
                                          plot=False)

    # check stratification
    c1 = splits['C1']
    c2 = splits['C2']
    c3 = splits['C3']
    train = splits['train']
    val = splits['val']


    check_split(train, val, c1, c2, c3, protein_col, substrate_col)
    logging.info("Split check passed.")

    dataset_len = len(metadata[[protein_col, substrate_col]].drop_duplicates())
    wandb.log({
        "dataset_len": dataset_len
    })
    # Create a table with columns: Subset, Class, Frequency
    table = wandb.Table(columns=["Subset", "Class", "Frequency"])

    for name, subset in [("Training", train), ("val", val), ("C1", c1), ("C2", c2), ("C3", c3)]:
        counts = subset["activity"].value_counts(normalize=True).sort_index()
        for cls, freq in counts.items():
            table.add_data(name, cls, freq)
        wandb.log({f"{name}/distribution_table": table})

    c1_emb = concatenated_embeddings[metadata.index.isin(c1.original_index)]
    c2_emb = concatenated_embeddings[metadata.index.isin(c2.original_index)]
    #c3_emb = concatenated_embeddings[metadata.index.isin(c3.original_index)]
    train_emb = concatenated_embeddings[metadata.index.isin(train.original_index)]
    val_emb = concatenated_embeddings[metadata.index.isin(val.original_index)]

    train_activity = activity[metadata.index.isin(train.original_index)]
    val_activity = activity[metadata.index.isin(val.original_index)]

    # Fit the model
    logging.info("Start training" + model_name + " with params :" + str(model_params))
    history = trainer.fit(train_emb, train_activity, val_emb, val_activity)

    # Log losses to wandb
    wandb.log({
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1] if history["val_loss"][-1] is not None else None
    })

    # Evaluate on sets
    results = {}
    return_metrics = 0.0
    for emb,name in [(train_emb,'train'), (val_emb,'val'),(c1_emb, "C1"), (c2_emb, "C2") ]:#,(c3_emb, "C3")
        y_pred = trainer.predict(emb)
        results[name] = y_pred

    test_sets = ["train", "val","C1","C2"]#,"C3"]
    metrics = [accuracy_score, roc_auc_score, f1_score, matthews_corrcoef]

    # Prepare a W&B Table
    pred_table = wandb.Table(
        columns=["Split", "Index", "y_true_raw", "y_true_bin", "y_pred_prob", "y_pred_bin"]
    )
    metrics_table = wandb.Table(columns=["Split", "Metric", "Value"])

    for split_name in test_sets:

        true_activities = metadata[metadata.index.isin(splits[split_name].original_index)]["activity"].values
        binary_true_activities = (true_activities != "none").astype(int)

        predicted_activities_prob = results[split_name]
        predicted_activities_bin = (predicted_activities_prob > 0.5).astype(int)

        for metric_fn in metrics:
            # Some metrics require probabilities (e.g., roc_auc_score)
            if metric_fn == roc_auc_score:
                value = metric_fn(binary_true_activities,predicted_activities_prob)  # pass probabilities
            else:
                value = metric_fn(binary_true_activities,predicted_activities_bin)  # pass binary predictions)
            if split_name == "val" and metric_fn == f1_score:
                return_metrics = value
            metrics_table.add_data(split_name, metric_fn.__name__, value)
        for idx, (raw, true, prob, pred) in enumerate(
            zip(true_activities, binary_true_activities, predicted_activities_prob, predicted_activities_bin)
        ):
            pred_table.add_data(split_name, idx, raw, int(true), float(prob), int(pred))
    # Log the table
    wandb.log({
        "metrics": metrics_table,
        "predictions": pred_table
    })
    run.finish()
    return return_metrics




def train_and_log(params: dict[str, Any] = None) -> None:
    logging.info("Starting experiment...")
    model_classes = params["models"]
    tasks = []
    for model_name, model_params in model_classes.items():
        logging.info(f"Preparing model: {model_name} with params: {model_params}")
        model_params_copy = model_params.copy()

        tasks.append(
            delayed(run_sklearn_experiment)(
                model_name=model_name,
                model_params=model_params_copy,
                substrate_name=params["substrate_name"],
                protein_name=params["protein_name"],
                project=params["project"],
                wandb_mode=params["wandb_mode"],
                sweep=False,
                concatenation_path=params.get("concatenation_path", None)
            )
        )
        try:
            with parallel_config(
                backend="loky",
                n_jobs=params["n_jobs"],
                max_nbytes=params["max_mem"],
            ):
                Parallel(n_jobs=params["n_jobs"], timeout=None)(tasks)
        except Exception:
            logging.error("Error occurred, attempting to clean up...")
            raise
        

def main():
    train_and_log(get_params("scikit_learn"))


if __name__ == "__main__":
    setup_logging()
    main()