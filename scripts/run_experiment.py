import logging
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Optional


from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

import wandb

from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier



from src.utils.helper_function import get_params, setup_logging, nano_id
from src.training.train import SklearnTrainer

MODEL_MAPPING = {
    "ridge_classifier": RidgeClassifier,
    "decision_tree" : DecisionTreeClassifier

    # add more models here
}

def run_sklearn_experiment(
    model_name: str,
    model_params: dict[str, Any],
    substrate_path:str,
    protein_path:str,
    substrate_name:str,
    protein_name:str,
    wandb_mode: str = "offline",
    project: str = "gt-substrate-predictor",
    sweep: bool = False,
) -> None:
    run = (
        wandb.init(
            project=project,
            name=f"{model_name}_substrate-{substrate_name}_protein-{protein_name}_id-{nano_id()}",
            # config={
            # },
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

    # --- Load your data here ---
    #TODO We need to implement this function. How do we get the data for the pipeline?
    X_train, y_train, X_val, y_val = [0,0,0,0]#load_data(substrate_path, protein_path)

    # Fit the model
    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=1)

    # Log losses to wandb
    wandb.log({
        "train_loss": history["train_loss"][-1],
        "val_loss": history["val_loss"][-1] if history["val_loss"][-1] is not None else None
    })

    run.finish()


def train(params: dict[str, Any] = None) -> None:

    logging.info("Starting experiment...")

    model_classes = params["models"]

    tasks = []
    for model_name, model_params in model_classes.items():
        for seed in tqdm(range(100), desc="generating seeds", leave=False):
            model_params_copy = model_params.copy()

            tasks.append(
                delayed(run_sklearn_experiment)(
                    model_name=model_name,
                    model_params=model_params_copy,
                    substrate_path=params["substrate_path"],
                    protein_path=params["protein_path"],
                    substrate_name=params["substrate_name"],
                    protein_name=params["protein_name"],
                    project=params["project"],
                    wandb_mode=params["wandb_mode"],
                    sweep=False,
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
        #gc.collect()
        raise


def main():
    train(get_params("benchmark"))


if __name__ == "__main__":
    setup_logging()
    main()