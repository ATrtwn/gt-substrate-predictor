"""
Train graph neural network for GT-substrate prediction.

Usage:
    python scripts/train_gnn.py
"""
import yaml
import logging
import sys
from pathlib import Path
import json
import argparse
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, HeteroData, Batch
import wandb
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.nn_model import GT_NN, BilinearInteractionNet, AttentionMLP, save_model
from src.models.gnn_models import GNN_3G_Classifier, MolecularEGNN_3G_Sparse
from src.data.data_split import stratified_split_by_entities, check_split
from src.utils.helper_function import get_params, setup_logging, nano_id
from datetime import datetime
from tqdm import tqdm
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler
# Ligand 3D generation moved to `src.data.ligand_structures` - this script
# will load cached ligand structures and compute donor ligand on-demand.
from src.data.ligand_structures import load_ligand_from_cache, compute_and_cache_ligand


# Feature helpers (loading and safe conversion) and atom features are provided
# by `src.features.feature_utils` and `src.data.preprocess_structure_prediction`.
# This script intentionally does not implement ligand construction — the
# computation was moved to `src.data.ligand_structures` to separate concerns.

def make_pair_heterodata(protein_data, ligand_data, y):
    """
    Combine protein and ligand `Data` objects into a HeteroData structure.

    Additionally, attach the ligand Data directly to the protein Data object as
    `protein_data.ligand` so it can be used when you want protein-only DataLoader
    that still carries the ligand information.
    """
    data = HeteroData()

    # ---- Protein graph ----
    data['protein'].x = protein_data.x
    data['protein'].edge_index = protein_data.edge_index
    data['protein'].pos = protein_data.pos
    if hasattr(protein_data, 'edge_attr'):
        data['protein'].edge_attr = protein_data.edge_attr

    # keep your metadata
    data['protein'].model_str = protein_data.model_str
    data['protein'].UGT_ID = protein_data.UGT_ID
    data['protein'].scalars = protein_data.scalars

    # ---- Ligand graph ----
    data['ligand'].x = ligand_data.x
    data['ligand'].edge_index = ligand_data.edge_index
    data['ligand'].pos = ligand_data.pos
    if hasattr(ligand_data, "edge_attr"):
        data['ligand'].edge_attr = ligand_data.edge_attr

    # ---- Label ----
    data.y = torch.as_tensor(y, dtype=torch.float)

    # Attach ligand object onto protein_data for convenience when using
    # DataLoader that yields `Data` objects (protein-centric)
    try:
        protein_data.ligand = ligand_data
        protein_data.y = torch.as_tensor(y, dtype=torch.float)
    except Exception:
        pass

    return data

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------ Protein graph lookup + dataset builder ------------------------


def find_protein_graph_by_UGT(UGT_ID: int, pocket_dir: Path):
    """Find and load the pocket graph corresponding to `UGT_ID`.

    This performs a direct file lookup by expected filename (<UGT_ID>_pocket_dataset.pt)
    and only loads that file, avoiding an expensive full-directory load.

    Returns a torch_geometric `Data` object or `None` if no graph found.
    """
    expected_name = f"{UGT_ID}_pocket_dataset.pt"
    expected_path = pocket_dir / expected_name
    if expected_path.exists():
        try:
            g = torch.load(expected_path, weights_only=False)
            # In case the file contains a list, pick the first matching object
            if isinstance(g, list):
                # find a Data-like object with the correct UGT_ID
                for item in g:
                    try:
                        if getattr(item, 'UGT_ID', None) == UGT_ID:
                            return item
                        if isinstance(item, dict) and item.get('UGT_ID') == UGT_ID:
                            return item
                    except Exception:
                        continue
                # fallback to first element
                return g[0] if len(g) > 0 else None
            # Data object or dict-like
            return g
        except Exception as e:
            print(f"Failed to load pocket graph {expected_path}: {e}")
            return None

    # Fallback: do a lightweight search through filenames to find any file containing UGT_ID
    try:
        for p in pocket_dir.iterdir():
            if p.suffix != '.pt':
                continue
            if f"{UGT_ID}_" in p.name or p.stem.startswith(str(UGT_ID)):
                try:
                    g = torch.load(p, weights_only=False)
                except Exception:
                    continue
                if isinstance(g, list):
                    for item in g:
                        try:
                            if getattr(item, 'UGT_ID', None) == UGT_ID:
                                return item
                            if isinstance(item, dict) and item.get('UGT_ID') == UGT_ID:
                                return item
                        except Exception:
                            continue
                    return g[0] if len(g) > 0 else None
                return g
    except Exception:
        print(f"Error while searching for UGT_ID {UGT_ID} in pocket graphs.")
        pass

    return None


def build_graphs_from_dataframe(df: pd.DataFrame, pocket_dir: Path, verbose: bool = True):
    """Return list of protein `Data` graphs (with .ligand and .y set) from a split DataFrame.

    Optimizations implemented:
     - Precompute unique ligand graphs (SMILES -> Data) once.
     - Sort rows by `UGT_ID` so the same protein is accessed repeatedly without reloading unnecessarily.
     - Cache loaded protein graphs and `deepcopy` a template per row to avoid mutating shared objects.

    The `verbose` flag enables additional diagnostics to explain why rows may be skipped.
    """
    graphs = []
    missed = 0

    # Basic sanity checks
    num_rows = len(df)
    num_smiles_nonnull = int(df['SMILES_isomeric_1'].dropna().shape[0]) if 'SMILES_isomeric_1' in df.columns else 0
    if verbose:
        print(f"Building graphs from DataFrame with {num_rows} rows ({num_smiles_nonnull} SMILES non-null); pocket_dir={pocket_dir}")

    # donor ligand (kept constant) — compute on-demand if not already cached
    donor_smiles = 'C1=CN(C(=O)NC1=O)[C@H]2[C@@H]([C@@H]([C@H](O2)COP(=O)(O)OP(=O)(O)O[C@@H]3[C@@H]([C@H]([C@@H]([C@H](O3)CO)O)O)O)O)O'
    donor = load_ligand_from_cache(donor_smiles)
    if donor is None:
        if verbose:
            print(f"[build_graphs_from_dataframe] Donor ligand not found in cache, attempting to compute: {donor_smiles}")
        donor = compute_and_cache_ligand(donor_smiles, force=False)
        if donor is None and verbose:
            print(f"[build_graphs_from_dataframe] Warning: donor ligand could not be computed: {donor_smiles}")

    # Precompute ligand graphs for unique SMILES in the dataframe (show progress)
    if 'SMILES_isomeric_1' not in df.columns:
        raise ValueError("DataFrame must contain column 'SMILES_isomeric_1'")

    unique_smiles = df['SMILES_isomeric_1'].dropna().unique()
    print("Starting ligand graph precomputation for", len(unique_smiles), "unique SMILES")
    lig_cache = {}
    missing_smiles = []
    for sm in tqdm(unique_smiles, desc='Precomputing ligand graphs'):
        lig = load_ligand_from_cache(sm)
        lig_cache[sm] = lig
        if lig is None:
            missing_smiles.append(sm)
    successful = sum(1 for v in lig_cache.values() if v is not None)
    print(f"Precomputed {successful}/{len(lig_cache)} unique ligand graphs")
    if missing_smiles and verbose:
        print(f"Warning: {len(missing_smiles)} SMILES missing cached 3D structures. Examples: {missing_smiles[:10]}")

    # Sort by UGT_ID to minimize repeated protein loads
    if 'UGT_ID' not in df.columns:
        raise ValueError("DataFrame must contain column 'UGT_ID'")
    df_sorted = df.sort_values('UGT_ID')

    # Keep only a single protein template (the current UGT) in memory at a time
    current_ugt = None
    current_prot_template = None
    missing_ugts = set()

    # Assemble dataset with progress bar; use itertuples for speed
    missing_rows_due_to_ligand = 0
    total_rows = len(df_sorted)
    for row in tqdm(df_sorted.itertuples(index=False), total=total_rows, desc='Assembling dataset'):
        UGT_ID = int(getattr(row, 'UGT_ID'))
        smiles = getattr(row, 'SMILES_isomeric_1')
        # Handle missing SMILES gracefully
        if pd.isna(smiles):
            missed += 1
            continue
        try:
            label = int(getattr(row, 'is_active'))
        except Exception:
            # Default to 0 if label missing/unparseable (won't stop build)
            label = 0

        # If we moved to a new UGT, release previous template to free memory
        if current_ugt != UGT_ID:
            # Explicitly delete previous reference
            if current_prot_template is not None:
                try:
                    del current_prot_template
                except Exception:
                    pass
            current_ugt = UGT_ID
            current_prot_template = find_protein_graph_by_UGT(UGT_ID, pocket_dir)
            if current_prot_template is None:
                missing_ugts.add(UGT_ID)

        prot_template = current_prot_template
        if prot_template is None:
            missed += 1
            continue

        lig = lig_cache.get(smiles)
        if lig is None:
            # record and skip: ligand structure not available
            missing_rows_due_to_ligand += 1
            missed += 1
            continue

        # Use a deep copy of the protein template to attach ligand and label
        try:
            prot = copy.deepcopy(prot_template)
            # keep consistent attribute names used elsewhere
            prot.acceptor = lig
            prot.donor = donor
            prot.y = torch.tensor(label, dtype=torch.float)
            graphs.append(prot)
        except Exception as e:
            missed += 1
            if verbose:
                print(f"[build_graphs_from_dataframe] Failed to attach ligand for UGT {UGT_ID}, SMILES {smiles}: {e}")
            continue

    if missing_rows_due_to_ligand > 0 and verbose:
        print(f"Skipped {missing_rows_due_to_ligand} rows because ligand 3D generation (cache) missing (see precompute warnings for examples)")
    if missing_ugts and verbose:
        print(f"Missing protein templates for {len(missing_ugts)} UGTs. Examples: {list(missing_ugts)[:10]}")

    print(f"Built dataset from DataFrame: {len(graphs)} graphs, missed {missed} rows")
    return graphs


def load_dataset_from_dataframe(df: pd.DataFrame, pocket_dir: Path, batch_size=16, shuffle=False, num_workers=4):
    """
    Build DataLoader from DataFrame containing rows with columns ['UGT_ID','SMILES_isomeric_1','is_active'].

    Returns a DataLoader of protein Data objects. Each `Data` object will have a `.ligand` attribute
    containing the ligand `Data` built from SMILES and `.y` attribute with the label.
    """
    graphs = build_graphs_from_dataframe(df, pocket_dir)
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


def train_epoch(model, train_loader, criterion, optimizer, device, noise_std=0.0):
    """
    Train GNN for one epoch (graph classification).

    This supports both single-graph models (that accept a single Batched `Data`)
    and three-graph models that expect a tuple/list: (protein_batch, sub1_batch, sub2_batch).

    Args:
        noise_std: Standard deviation for Gaussian noise augmentation on node features (0.0 = no noise)
    """
    model.train()
    total_loss = 0

    for batch in train_loader:
        # `batch` is a Batched protein Data; each original protein Data is expected to have
        # attached ligand Data objects (e.g. `.acceptor`, `.donor` or `.ligand`).
        # We build separate Batched ligand inputs for models that expect a tuple.
        prot_batch = batch.to(device)

        # Prepare ligand batches from the per-graph Data objects
        data_list = prot_batch.to_data_list()
        lig1_list = []
        lig2_list = []
        for g in data_list:
            lig1 = getattr(g, 'acceptor', None) or getattr(g, 'ligand', None) or getattr(g, 'substrate1', None) or getattr(g, 'ligand1', None)
            lig2 = getattr(g, 'donor', None) or getattr(g, 'ligand', None) or getattr(g, 'substrate2', None) or getattr(g, 'ligand2', None)
            if lig1 is None:
                raise ValueError('Missing ligand (acceptor) on a Data object in batch. Ensure dataset provides per-graph ligand Data.')
            if lig2 is None:
                # duplicate if donor missing
                lig2 = lig1
            lig1_list.append(lig1)
            lig2_list.append(lig2)

        s1_batch = Batch.from_data_list(lig1_list).to(device)
        s2_batch = Batch.from_data_list(lig2_list).to(device)

        labels = prot_batch.y

        # Optional Gaussian noise on node features (apply to protein and ligands)
        if noise_std > 0:
            prot_batch.x = prot_batch.x + torch.randn_like(prot_batch.x) * noise_std
            s1_batch.x = s1_batch.x + torch.randn_like(s1_batch.x) * noise_std
            s2_batch.x = s2_batch.x + torch.randn_like(s2_batch.x) * noise_std

        optimizer.zero_grad()

        # Call model with either single-batch or 3-tuple depending on model.forward
        #try:
        outputs = model([prot_batch, s1_batch, s2_batch])  # shape: [num_graphs, num_classes]
        # except Exception:
        #     # Fallback for models that accept a single batched Data
        #     outputs = model(prot_batch)

        labels = labels.view(-1,1).float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """Evaluate GNN model on validation/test set (graph classification)."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            # Batch is a Batched protein Data with per-graph ligand Data attached
            prot_batch = batch.to(device)

            # Build ligand batches
            data_list = prot_batch.to_data_list()
            lig1_list = []
            lig2_list = []
            for g in data_list:
                lig1 = getattr(g, 'acceptor', None) or getattr(g, 'ligand', None) or getattr(g, 'substrate1', None) or getattr(g, 'ligand1', None)
                lig2 = getattr(g, 'donor', None) or getattr(g, 'ligand', None) or getattr(g, 'substrate2', None) or getattr(g, 'ligand2', None)
                if lig1 is None:
                    raise ValueError('Missing ligand (acceptor) on a Data object in batch. Ensure dataset provides per-graph ligand Data.')
                if lig2 is None:
                    lig2 = lig1
                lig1_list.append(lig1)
                lig2_list.append(lig2)

            s1_batch = Batch.from_data_list(lig1_list).to(device)
            s2_batch = Batch.from_data_list(lig2_list).to(device)

            labels = prot_batch.y
            labels = labels.view(-1,1).float()

            # Forward pass (graph-level classification)
            try:
                outputs = model([prot_batch, s1_batch, s2_batch])  # shape: [num_graphs, num_classes]
            except Exception:
                # Fallback for single-graph models
                outputs = model(prot_batch)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get probabilities if binary classification (num_classes=2)
            if outputs.size(1) == 1:
                probs = torch.sigmoid(outputs)
            else:
                probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
    if len(data_loader)>0:
        avg_loss = total_loss / len(data_loader)
    else:
        avg_loss = 0
    if len(all_preds)>0:
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
    else:
        all_preds = np.empty((0,1),dtype=np.float32)
        all_labels = np.empty((0,1),dtype=np.float32)
    return avg_loss, all_preds, all_labels


def train_gnn_experiment(
    model_type: str,
    dataset_path: str,
    hidden_dims: list,
    dropout: float,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    oversample:bool,
    weight_decay: float = 0.0,
    num_heads: int = 4,
    num_workers: int = 4,
    use_residual: bool = True,
    data_augmentation: bool = False,
    shufle_train: bool = True,
    noise_std: float = 0.02,
    wandb_mode: str = "offline",
    project: str = "gt-substrate-predictor",
    optimizer_name: str = "adam",
    scheduler_type: str = "reduce_on_plateau",
    momentum: float = 0.9,
    step_size: int = 20,
    gamma: float = 0.1,
    activation: str = "relu",
    seed: int = None,
    save_path: str = None,
):
    """
    Train neural network experiment.
    
    Args:
        data_augmentation: Enable Gaussian noise augmentation during training
        noise_std: Standard deviation of Gaussian noise (default: 0.02)
        label_smoothing: Label smoothing factor (0.0 = disabled, 0.1-0.2 recommended)
        seed: Random seed for reproducibility (for ensemble training)
        save_path: Custom path to save the model (for ensemble training)
    """
    def load_trainset(df, name, shuffle=False, oversample = False):
        # If a DataFrame is passed in place of filename, build dataset from it
        pocket_dir = Path(__file__).resolve().parent.parent.parent / dataset_path
        if isinstance(df, pd.DataFrame):
            graphs = build_graphs_from_dataframe(df, pocket_dir)
            print(f"Built training set from DataFrame with {len(graphs)} objects")
        else:
            path = Path(__file__).resolve().parent.parent.parent / dataset_path / name
            graphs = torch.load(path, weights_only=False)
            print(f"Loaded training set with {len(graphs)} objects")
            try:
                print(f"DEBUG : Graph Keys founf : {graphs[0].keys()}")
            except Exception:
                pass
        if oversample:
            labels = torch.tensor([int(g.y.item()) for g in graphs])
            class_counts = Counter(labels.tolist())

            if len(class_counts) == 2:
                n_neg = class_counts[0]
                n_pos = class_counts[1]

                weights = {
                    0: 1.0 / n_neg,
                    1: 1.0 / n_pos
                }
                sample_weights = torch.tensor(
                    [weights[int(y.item())] for y in labels],
                    dtype= torch.double
                )
                sampler = WeightedRandomSampler(
                    weights = sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True
                )
                print(f"Oversampling enabled | Class dict | {dict(class_counts)}")
                return DataLoader(
                    graphs,
                    batch_size=batch_size,
                    sampler=sampler,
                    shuffle = False,
                    num_workers = num_workers,
                    pin_memory=True
                )
            else:
                print("Oversampling skipped - dataset is not binary")
        return load_dataset(df, name,shuffle=shuffle)

    def load_dataset(df,name, shuffle=False):
        # Support passing a DataFrame to build the graphs on-the-fly
        pocket_dir = Path(__file__).resolve().parent.parent.parent / dataset_path
        if isinstance(df, pd.DataFrame):
            graphs = build_graphs_from_dataframe(df, pocket_dir)
            print(f"Built {name} dataset from DataFrame: {len(graphs)} graphs")
        else:
            path = Path(__file__).resolve().parent.parent.parent / dataset_path / name
            print(f"Loading {name} from {path}")
            graphs = torch.load(path,weights_only=False)
            print(f"  -> {len(graphs)} graphs loaded")

        return DataLoader(
            graphs,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    # Load precomputed splits from CSVs
    split_path = Path(__file__).resolve().parent.parent / "data"
    split_df = pd.read_csv(split_path / "split.csv")
    train = split_df[split_df['split']=='train']
    val1 = split_df[split_df['split']=='val_C1']
    val2 = split_df[split_df['split']=='val_C2']
    val3 = split_df[split_df['split']=='val_C3']
    val = split_df[split_df['split'].isin(['val_C1', 'val_C2', 'val_C3'])]
    c1_test = split_df[split_df['split']=='test_C1']
    c2_test = split_df[split_df['split']=='test_C2']
    c3_test = split_df[split_df['split']=='test_C3']
    df_list = {
        "train": train,
        "val_C1": val1,
        "val_C2": val2, 
        "val_C3": val3,
        "val": val,
        "test_C1": c1_test,
        "test_C2": c2_test,
        "test_C3": c3_test
    }

    # Create small temporary datasets (first 10 rows) for quick tests and add them to df_list
    for name, subdf in list(df_list.items()):
        if subdf is None or len(subdf) == 0:
            continue
        df_list[name] = subdf.head(300)
    
    # Set random seed if provided
    if seed is not None:
        set_seed(seed)
        logging.info(f"Random seed set to: {seed}")
    
    # Initialize W&B
    run_name = f"{model_type}_boltz_protein_structure"
    if seed is not None:
        run_name += f"_seed-{seed}"
    # Use current date + minute-of-day as a stable, human-readable run id
    now = datetime.now()
    minute_of_day = now.hour * 60 + now.minute
    file_id = f"{now.strftime('%Y%m%d')}_{minute_of_day}"
    run_name += f"_id-{file_id}"
    
    run = wandb.init(
        project=project,
        name=run_name,
        config={
            "model": model_type,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "weight_decay": weight_decay,
            "activation": activation,
        },
        mode=wandb_mode,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    loaders = {}
    #Create DataLoaders
    for name in df_list.keys():
        shuffle = (name == "train") and shufle_train
        if name == "train":
            loaders[name] = load_trainset(df_list[name],name,shuffle,oversample=oversample)
        else:
            loaders[name] = load_dataset(df_list[name], name, shuffle)

    num_scalar_features = loaders["train"].dataset[0].scalars.shape[0]
    # Apply scaling to all graphs in train, val, test datasets

    # Initialize model
    print(f"DEBUG : Hidden dims {hidden_dims}")
    print(f"DEBUG : num scalar features {num_scalar_features}")

    # Derive node/feature dims from first training example
    prot_in = loaders["train"].dataset[0].num_node_features
    lig_in = getattr(loaders["train"].dataset[0].acceptor, 'num_node_features', prot_in)
    donor_in = getattr(loaders["train"].dataset[0].donor, 'num_node_features', lig_in)
    ligand_scalar_dim = 0
    try:
        lig_scalars = getattr(loaders["train"].dataset[0].acceptor, 'scalars', None) or getattr(loaders["train"].dataset[0].acceptor, 'scalar_feats', None)
        if lig_scalars is not None:
            ligand_scalar_dim = lig_scalars.shape[0]
    except Exception:
        ligand_scalar_dim = 0

    # Normalize hidden dims for EGNN constructor
    if isinstance(hidden_dims, (list, tuple)):
        egnn_hidden_dim = hidden_dims[-1]
    else:
        egnn_hidden_dim = hidden_dims

    # Instantiate appropriate model with correct parameter names
    if model_type in ["Transformer", "GINE", "GATv2", "GAT", "GIN", "GraphSAGE"]:
        model = GNN_3G_Classifier(
            protein_in_channels=prot_in,
            ligand_in_channels=lig_in,
            ligand1_in_channels=lig_in,
            ligand2_in_channels=donor_in,
            hidden_channels=hidden_dims,
            dropout=dropout,
            num_classes=1,  # Binary classification (BCEWithLogits)
            layer_name=model_type,
            heads=num_heads,
            use_residual=use_residual,
            protein_scalar_dim=num_scalar_features,
            ligand_scalar_dim=ligand_scalar_dim,
            attn_heads=num_heads,
            concat=True,
        ).to(device)
    elif model_type == "MolecularEGNN":
        model = MolecularEGNN_3G_Sparse(
            ligand_in_channels=lig_in,
            ligand1_in_channels=lig_in,
            ligand2_in_channels=donor_in,
            in_dim=egnn_hidden_dim,
            hidden_dim=egnn_hidden_dim,
            embedding_size=32,
            depth=4,
            num_classes=1,
            dropout=dropout,
            protein_scalar_dim=num_scalar_features,
            ligand_scalar_dim=ligand_scalar_dim,
            attn_heads=num_heads,
            edge_attr_dim=11,
            m_dim=64,
            use_residual=use_residual,
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type for 3G models: {model_type}")
    train_labels = []
    # Calculate class weights for imbalanced data
    for batch in loaders["train"]:
        # batch.y shape: [num_graphs] for graph classification
        train_labels.append(batch.y.cpu())

    train_labels = torch.cat(train_labels).numpy().astype(int)
    # Convert to int for bincount (use original binary labels before smoothing)
    train_labels_int = np.round(train_labels).astype(int)
    if oversample:
        criterion = nn.BCEWithLogitsLoss()
    else:
        class_counts = np.bincount(train_labels_int)
        n_neg = class_counts[0]
        n_pos = class_counts[1]
        if n_pos==0:
            raise ValueError("No positive samples in training set")
        pos_weight = torch.tensor([n_neg / n_pos], device = device, dtype=torch.float32)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Create optimizer based on config
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Create learning rate scheduler
    if scheduler_type.lower() == 'reduce_on_plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    elif scheduler_type.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type.lower() == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    logging.info(f"Optimizer: {optimizer_name}")
    logging.info(f"Scheduler: {scheduler_type}")
    logging.info(f"Weight decay (L2 reg): {weight_decay}")
    
    logging.info(f"Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10
    
    # Track metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_roc_aucs = []
    
    # Determine noise level for augmentation
    augmentation_noise = noise_std if data_augmentation else 0.0
    if data_augmentation:
        logging.info(f"Data augmentation enabled: Gaussian noise with std={noise_std}")
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, loaders["train"], criterion, optimizer, device, noise_std=augmentation_noise)
        
        # Evaluate on VALIDATION set for early stopping (proper ML practice)
        val_loss, val_preds, val_true = evaluate(model, loaders["val"], criterion, device)
        
        # Convert smoothed labels back to binary for evaluation
        val_true_binary = np.round(val_true).astype(int)
        
        # Metrics
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_acc = accuracy_score(val_true_binary, val_preds_binary)
        val_f1 = f1_score(val_true_binary, val_preds_binary)
        val_roc_auc = roc_auc_score(val_true_binary, val_preds)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_roc_aucs.append(val_roc_auc)
        
        # Log to W&B
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "val_roc_auc": val_roc_auc,
            "learning_rate": optimizer.param_groups[0]['lr']
        })
        
        logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            if scheduler_type.lower() == 'reduce_on_plateau':
                scheduler.step(val_loss)
            else:  # step or cosine
                scheduler.step()
        
        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            # Save best model (with seed suffix if provided)
            if save_path is not None:
                model_path = save_path
            elif seed is not None:
                model_path = f"experiments/best_model_{model_type}_seed_{seed}.pth"
            else:
                model_path = f"experiments/best_model_{model_type}.pth"
            save_model(model, optimizer, epoch, val_loss, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model state before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logging.info("Loaded best model state for final evaluation")
    
    # Plot training curves
    plot_dir = Path("reports/figures")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Graph Neural Network Training - {model_type}', fontsize=14, fontweight='bold')
    
    # Loss curves
    axes[0, 0].plot(train_losses, label='Train Loss', linewidth=2)
    axes[0, 0].plot(val_losses, label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(val_accuracies, label='Val Accuracy', color='green', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 0].plot(val_f1_scores, label='Val F1', color='orange', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('Validation F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # ROC-AUC
    axes[1, 1].plot(val_roc_aucs, label='Val ROC-AUC', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC')
    axes[1, 1].set_title('Validation ROC-AUC')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plot_dir / f"gnn_training_{model_type}_{file_id}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Training curves saved to {plot_path}")
    
    # Final evaluation on test sets
    logging.info("Evaluating on test sets...")
    
    # Record dataset sizes (safe access in case some loaders are None)
    train_size = len(loaders["train"].dataset) if loaders.get("train") is not None else 0
    val_c1_size = len(loaders["val_C1"].dataset) if loaders.get("val_C1") is not None else 0
    val_c2_size = len(loaders["val_C2"].dataset) if loaders.get("val_C2") is not None else 0
    val_c3_size = len(loaders["val_C3"].dataset) if loaders.get("val_C3") is not None else 0
    val_size = len(loaders["val"].dataset) if loaders.get("val") is not None else 0
    test_c1_size = len(loaders["test_C1"].dataset) if loaders.get("test_C1") is not None else 0
    test_c2_size = len(loaders["test_C2"].dataset) if loaders.get("test_C2") is not None else 0
    test_c3_size = len(loaders["test_C3"].dataset) if loaders.get("test_C3") is not None else 0

    results_metrics = {
        "model": model_type,
        "hidden_dims": hidden_dims,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs_trained": len(train_losses),
        "train_set_size": train_size,
        "val_C1_size": val_c1_size,
        "val_C2_size": val_c2_size,
        "val_C3_size": val_c3_size,
        "val_size": val_size,
        "test_C1_size": test_c1_size,
        "test_C2_size": test_c2_size,
        "test_C3_size": test_c3_size,
        "best_val_loss": float(best_val_loss),
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss": float(val_losses[-1]),
    }
    
     # Evaluate on all three test sets: C1, C2, and C3
    for split_name, split_loader in [
        ("C1", loaders["test_C1"]),
        ("C2", loaders["test_C2"]),
        ("C3", loaders["test_C3"])
    ]:
        if len(loaders["test_C3"])>0:
            _, test_preds, test_labels = evaluate(model, split_loader, criterion, device)
            test_preds_binary = (test_preds > 0.5).astype(int)

            # Convert smoothed labels back to binary for evaluation
            test_labels_binary = np.round(test_labels).astype(int)

            # Calculate metrics
            acc = accuracy_score(test_labels_binary, test_preds_binary)
            f1 = f1_score(test_labels_binary, test_preds_binary)
            roc_auc = roc_auc_score(test_labels_binary, test_preds)
            mcc = matthews_corrcoef(test_labels_binary, test_preds_binary)
            #Bootrap standard error
            rng = np.random.default_rng(42)
            boots_idx = [rng.integers(0,len(test_labels_binary),len(test_labels_binary)) for _ in range(1000)]
            f1_se = np.std(
                [f1_score(
                        test_labels_binary[i],test_preds_binary[i]
                    ) for i in boots_idx
                ],ddof=1
            )
            acc_se = np.std(
                [accuracy_score(
                    test_labels_binary[i],test_preds_binary[i]
                    ) for i in boots_idx
                ],ddof=1
            )
            roc_auc_se = np.std(
                [roc_auc_score(
                    test_labels_binary[i],test_preds_binary[i]
                    ) for i in boots_idx
                ],ddof=1
            )
            mse_se = np.std(
                [matthews_corrcoef(
                        test_labels_binary[i],test_preds_binary[i]
                        )for i in boots_idx
                ],ddof=1
            )
            wandb.log({
                f"{split_name}/accuracy": acc,
                f"{split_name}/f1": f1,
                f"{split_name}/roc_auc": roc_auc,
                f"{split_name}/mcc": mcc,
            })

            # Store test results
            results_metrics[f"{split_name}_accuracy"] = float(acc)
            results_metrics[f"{split_name}_f1"] = float(f1)
            results_metrics[f"{split_name}_roc_auc"] = float(roc_auc)
            results_metrics[f"{split_name}_mcc"] = float(mcc)
            results_metrics[f"{split_name}_f1_se"] = float(f1_se)
            results_metrics[f"{split_name}_acc_se"] = float(acc_se)
        
            logging.info(f"{split_name} - Acc: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}, MCC: {mcc:.4f}")
    
    # Save metrics to JSON
    results_dir = Path("reports/metrics")
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"gnn_metrics_{model_type}_{file_id}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results_metrics, f, indent=2)
    
    logging.info(f"Metrics saved to {results_path}")
    
    run.finish()
    logging.info("Training complete!")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train graph neural network for GT-substrate prediction')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (for ensemble)')
    parser.add_argument('--save_path', type=str, default=None, help='Custom path to save model (for ensemble)')
    args = parser.parse_args()
    
    params = get_params("graph_neural_network")
    for model in params["model_type"]: 
        train_gnn_experiment(
            dataset_path=Path(params["dataset_path"]),
            model_type=model,
            hidden_dims=params["hidden_dims"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            oversample=params["oversample"],
            weight_decay=params.get("weight_decay", 0.0),
            num_heads=params.get("num_heads", 4),
            num_workers=params.get("num_workers", 4),
            use_residual=params.get("use_residual", True),
            data_augmentation=params.get("data_augmentation", False),
            shufle_train=params.get("shuffle_train", True),
            noise_std=params.get("noise_std", 0.02),
            wandb_mode=params["wandb_mode"],
            project=params["project"],
            optimizer_name=params.get("optimizer", "adam"),
            scheduler_type=params.get("scheduler", "reduce_on_plateau"),
            momentum=params.get("momentum", 0.9),
            step_size=params.get("step_size", 20),
            gamma=params.get("gamma", 0.1),
            activation=params.get("activation", "relu"),
            seed=args.seed,
            save_path=args.save_path,
        )


if __name__ == "__main__":
    setup_logging()
    main()
