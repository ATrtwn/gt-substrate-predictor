import sys
import wandb
import optuna
import json
import logging

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
data_dir = Path(__file__).parent.parent 

from src.data.data_split import stratified_split_by_entities, check_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef

import numpy as np
import pandas as pd

from src.utils.helper_function import get_params, setup_logging, nano_id, sample_params
from src.training.train import SklearnTrainer

from datetime import datetime
from typing import Any, Optional

from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

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

# Define sweep config that Optuna will control
sweep_config = {
    "method": "bayes",  # Wandb's bayesian optimization (not used directly)
    "metric": {
        "name": "val_loss",
        "goal": "minimize"
    },
    "parameters": {
        # These will be overridden by Optuna
        "lr": {"values": [0.001, 0.01]},
        "batch_size": {"values": [32, 64]}
    }
}

class OptunaWandbSweep:
    def __init__(self, wandb_par:dict=None, paths:dict=None, optuna_config:dict=None, param_space:dict=None):
        self.wandb_par = wandb_par
        self.optuna_config = optuna_config
        self.param_space = param_space
        self.paths = paths
        

    def run_sklearn_experiment(
        self,
        model_name: str,
        model_params: dict[str, Any],
        substrate_name:str,
        protein_name:str,
        wandb_mode: str = "offline",
        concatenation_path:str=None,
    ) -> float:

        log_dict = {}
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

        log_dict["unique_proteins"] = len(unique_proteins)
        log_dict["unique_substrates"] = len(unique_substrates)

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

        dataset_len = len(metadata[[protein_col, substrate_col]].drop_duplicates())
        
        log_dict["dataset_len"] = dataset_len

        # Create a table with columns: Subset, Class, Frequency
        table = wandb.Table(columns=["Subset", "Class", "Frequency"])

        for name, subset in [("Training", train), ("val", val), ("C1", c1), ("C2", c2), ("C3", c3)]:
            counts = subset["activity"].value_counts(normalize=True).sort_index()
            for cls, freq in counts.items():
                table.add_data(name, cls, freq)

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
        
        log_dict["train_loss"] = history["train_loss"][-1]
        log_dict["val_loss"] = history["val_loss"][-1] if history["val_loss"][-1] is not None else None

        # Evaluate on sets
        results = {}
        val_loss = 0.0
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
                    val_loss = value
                metrics_table.add_data(split_name, metric_fn.__name__, value)
            for idx, (raw, true, prob, pred) in enumerate(
                zip(true_activities, binary_true_activities, predicted_activities_prob, predicted_activities_bin)
            ):
                pred_table.add_data(split_name, idx, raw, int(true), float(prob), int(pred))
        # Log the table
        
        log_dict["metrics_table"] = metrics_table
        log_dict["predictions_table"] = pred_table
        log_dict["val_loss"] = val_loss
        return log_dict
    
    def objective(self, trial: optuna.Trial):
        # Get parameters from Optuna
        selected_params = sample_params(trial, param_space=self.param_space)
        # Initialize wandb with these parameters
        
        # Your training logic
        log_dict = self.run_sklearn_experiment(
            model_name=self.optuna_config["model_name"],
            model_params=selected_params,
            substrate_name=self.paths["substrate_name"],
            protein_name=self.paths["protein_name"],
            wandb_mode=self.wandb_par["offline"],
            concatenation_path=self.paths["concatenation_path"],
        )
        val_loss = log_dict["val_loss"]
        
        # Log metrics

        trial.set_user_attr("val_loss", log_dict["val_loss"])
        trial.set_user_attr("unique_proteins", log_dict["unique_proteins"])
        trial.set_user_attr("unique_substrates", log_dict["unique_substrates"])
        trial.set_user_attr("dataset_len", log_dict["dataset_len"])
        trial.set_user_attr("train_loss", log_dict["train_loss"])
        trial.set_user_attr("predictions_table", log_dict["predictions_table"])
        trial.set_user_attr("metrics_table", log_dict["metrics_table"])
        
        return val_loss
    

    def run_sweep(self):
        # Create Optuna study
        study = optuna.create_study(
            direction=self.optuna_config["direction"],
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Run optimization
        study.optimize(self.objective, n_trials=self.optuna_config["n_trials"],show_progress_bar=True)


        completed_trials = [
            t for t in study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]

        if not completed_trials:
            print("⚠️ No completed trials found!")
            return study
        
       # Log best 5 hyperparameters to wandb
       #we want to maximize so we reverse sort
        best_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
        for i, trial in enumerate(best_trials):
            wandb.init(
                project=self.wandb_par["project"],
                name=f"best_trial_{i+1}_{self.optuna_config["model_name"]}_{nano_id(6)}",
                config=trial.params,
                reinit=True
            )
            wandb.run.summary["val_loss"] = trial.value
            wandb.run.summary["unique_proteins"] = trial.user_attrs["unique_proteins"]
            wandb.run.summary["unique_substrates"] = trial.user_attrs["unique_substrates"]
            wandb.run.summary["dataset_len"] = trial.user_attrs["dataset_len"]
            wandb.run.summary["train_loss"] = trial.user_attrs["train_loss"]
            wandb.log({
                "predictions_table": trial.user_attrs["predictions_table"],
                "metrics_table": trial.user_attrs["metrics_table"]
            })
        return study

# Usage

def main():
    params = get_params("scikit_learn_opt")
    sweep = OptunaWandbSweep(
        wandb_par = params["wandb"],
        paths = params["paths"],
        optuna_config = {
            "study_name": params["optuna_config"]["study_name"],
            "direction": params["optuna_config"]["direction"],
            "n_trials": params["optuna_config"]["n_trials"],
            "n_jobs": params["optuna_config"]["n_jobs"],
            "model_name": params["optuna_config"]["model_name"]
        },
        param_space = params["optuna_config"]["param_space"]
    )
    sweep.run_sweep()


if __name__ == "__main__":
    setup_logging()
    main()