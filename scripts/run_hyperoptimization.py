import sys
import wandb
import optuna
import json
import logging
import os

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
from sklearn.preprocessing import StandardScaler
ROOT = Path(__file__).parent.parent


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
        metrics_list = []
        pred_data = []
        return_f1 = 0.0
        val_loss = history["val_loss"][-1] if history["val_loss"] else None
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
        log_dict = {}
        log_dict["unique_proteins"] = metadata['UGT_ID'].nunique()
        log_dict["unique_substrates"] = metadata['substrate'].nunique()
        log_dict["dataset_len"] = len(metadata)
        log_dict["metrics_table"] = metrics_df
        log_dict["predictions_table"] = preds_df
        log_dict["val_loss"] = val_loss
        log_dict["train_counts"] = np.bincount(train_labels).tolist()
        log_dict["val_counts"] = np.bincount(val_labels).tolist()
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
        trial.set_user_attr("train_counts", log_dict["train_counts"] if "train_counts" in log_dict else None)
        trial.set_user_attr("dataset_len", log_dict["dataset_len"])
        trial.set_user_attr("train_loss", log_dict["train_loss"])
        trial.set_user_attr("predictions_table", log_dict["predictions_table"])
        trial.set_user_attr("metrics_table", log_dict["metrics_table"])
        
        return val_loss
    

    def run_sweep(self):
        # 1. Added SQLite storage to save Optuna progress to a local file
        study_db = f"sqlite:///{self.optuna_config.get('study_name', 'study')}.db"
        
        study = optuna.create_study(
            study_name=self.optuna_config["study_name"],
            storage=f"sqlite:///{self.optuna_config['study_name']}.db", # Local file
            load_if_exists=True,                                       # Resume if interrupted
            direction=self.optuna_config["direction"],
            pruner=optuna.pruners.MedianPruner(),
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(self.objective, n_trials=self.optuna_config["n_trials"], show_progress_bar=True)

        completed_trials = [
            t for t in study.trials 
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]

        if not completed_trials:
            print("⚠️ No completed trials found!")
            return study
        
        # Determine if we maximize or minimize based on config
        is_reverse = True if self.optuna_config["direction"] == "maximize" else False
        best_trials = sorted(completed_trials, key=lambda t: t.value, reverse=is_reverse)[:5]

        # 1. Create the output directory
        output_dir = ROOT /"reports" / "results_scikit_learn" / f"{self.optuna_config['study_name']}_substrate-{self.paths['substrate_name']}_protein-{self.paths['protein_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(output_dir, exist_ok=True)

        # 2. Export All Trials Comparison
        # This converts the study to a dataframe, including your user_attrs
        df = study.trials_dataframe()
        
        # Rename columns for readability (Optuna prefixes them with 'user_attrs_')
        df.columns = [c.replace('user_attrs_', '') if 'user_attrs_' in c else c for c in df.columns]
        
        comparison_path = os.path.join(output_dir, f"{self.optuna_config['study_name']}_comparison.csv")
        df.to_csv(comparison_path, index=False)

        # 3. Export Best Trial Detailed Metrics
        best_trial = study.best_trial
        best_results = {
            "best_trial_number": best_trial.number,
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "all_metrics": {
                # Extract specific attributes you logged in objective()
                "val_loss": best_trial.user_attrs.get("val_loss"),
                "train_loss": best_trial.user_attrs.get("train_loss"),
                "dataset_len": best_trial.user_attrs.get("dataset_len"),
                # Exclude large objects like tables from the JSON summary if necessary
            }
        }

        best_json_path = os.path.join(output_dir, f"best_model_{self.optuna_config['model_name']}.json")
        with open(best_json_path, "w") as f:
            json.dump(best_results, f, indent=4)

        print(f"📊 Reports generated in {output_dir}")
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