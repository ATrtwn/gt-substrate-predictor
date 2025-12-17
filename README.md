# gt-substrate-predictor

## 🧬 Predicting Glycosyltransferase-Substrate Pairs Using Machine Learning

### Project Overview
Glycosylation is a fundamental biochemical process where a sugar is transferred to another metabolite by glycosyltransferase (GT) enzymes. Predicting which GT can glycosylate a specific substrate is challenging due to variability in acceptor-binding pockets. 

This project aims to develop machine learning models that predict GT-substrate binding using protein and substrate embeddings, with optional integration of AlphaFold2 structural features. The goal is to enable targeted modification of bioactive metabolites and pharmaceuticals.

---

### 🎯 Key Objectives
- Build representations of GT proteins (ProtT5, ProstT5) and substrates (ChemBERTa, KPGT).
- Train and evaluate ML models (Logistic Regression, FNN, GNN) to classify GT-substrate pairs.
- Evaluate models on stratified splits:
  - **C1:** Both GT and substrate seen during training
  - **C2:** Novel GT or substrate
  - **C3:** Both GT and substrate unseen
- Benchmark against baseline models and existing prediction tools.
- Integrate structural insights from AlphaFold2 (optional).

---

### 📂 Repository Structure
```
project_root/
│
├── data/ # Raw and processed datasets
├── scripts/ # Entry-point scripts
├── src/ # Source code for data processing, embeddings, models, training, and evaluation
│ ├── data/ # Data loading, preprocessing, and splitting
│ ├── features/ # Feature extraction and encoding (e.g., embeddings)
│ ├── models/ # Model architectures
│ ├── training/ # Training loops, optimizers, evaluation
│ └── utils/ # Shared helper functions (e.g. plotting)
├── experiments/ # Experiment logs and checkpoints
└── reports/ # Figures and summaries of results
```

### ⚡ Usage

#### 1. Generate CSV files from the Access database

1. Create a `.env` file in the project root with:

    - ACCESS_DB_PATH=/full/path/to/database.accdb
    - ACCESS_DB_PASSWORD=yourpassword

2. Run the script:

   ```bash
   python scripts/fetch_data.py
   ```
    
   The script will:
   - Check if UGT.csv, Activity.csv, and Substrate.csv already exist in data/
   - If missing, export the tables from the .accdb file and fetch additional substrate info

3. Result: CSV files will be saved in the data/ folder, ready for preprocessing and analysis.

#### 2. Generate Concatenated Embeddings

After generating protein embeddings (ProtT5) and substrate embeddings (ChemBERTa2, ChemBERTa3, KPGT), concatenate them for ML model training:

**Generate all substrate embedding types:**
```bash
python scripts/concatenate_embeddings.py --substrate all
```

**Generate specific substrate type only:**
```bash
# ChemBERTa2 (384D) + ProtT5 (1024D) = 1408D
python scripts/concatenate_embeddings.py --substrate chemberta2

# ChemBERTa3 (768D) + ProtT5 (1024D) = 1792D
python scripts/concatenate_embeddings.py --substrate chemberta3

# KPGT (2304D) + ProtT5 (1024D) = 3328D
python scripts/concatenate_embeddings.py --substrate kpgt
```
3. Run the clustering:
-Once MMseqs2 is ready, run the clustering command (adjust filenames if needed):
´´´powershell tools\mmseqs\bin\mmseqs.exe easy-cluster UGT.fasta GT_cluster tmp --min-seq-id 0.7 -c 0.7´´´
  -min-seq-id 0.7 sets 70% minimum sequence identity (agreed on the meeting)
  -c 0.7 sets 70% minimum coverage (share ≥70% of their length)

**Output:**
- `data/concatenated_embeddings/X_{substrate_type}.npy` - Concatenated embeddings (N, dim)
- `data/concatenated_embeddings/y_{substrate_type}.npy` - Activity labels (N,)
- `data/concatenated_embeddings/metadata_{substrate_type}.csv` - Protein names, substrate names, indices

The script automatically:
- Maps protein-substrate pairs from `Activity.csv`
- Only includes pairs with both protein and substrate embeddings
- Generates 2251 valid concatenated pairs (100% coverage)

5. Report output:
  -´´´powershell python .\scripts\print_cluster_report.p´´´ for the report output -> CONCLUSION: dataset is diverse enough, no need for omiting the sequences

#### 🧬 Substrate embeddings
To use RDKit in this project, follow these steps:
1. Create and activate a Conda environment
conda create -n fast_env python=3.11
conda activate fast_env
2. Install RDKit: mamba install -c conda-forge rdkit
3. Configure VS Code: Open your project in VS Code.
  Press Ctrl+Shift+P → Python: Select Interpreter → choose the Python from fast_env.
  Open a terminal in VS Code and make sure it shows: (fast_env) PS C:\path\to\project>
4. Test the installation by runing the following command in the VS Code terminal or Anaconda Prompt: python -c "from rdkit import Chem; mol = Chem.MolFromSmiles('C1CCCCC1'); print(mol)"
  Expected output: <rdkit.Chem.rdchem.Mol object at 0x...>
Tips:
  Always activate fast_env before running scripts or installing additional packages.
  Selecting the correct interpreter in VS Code ensures your scripts use the environment where RDKit is installed.
  After this setup, RDKit can be used seamlessly in Python scripts and notebooks within this project.
MODEL: DeepChem/ChemBERTa-2_MTR (Trained on masked-token prediction + molecular property tasks): 
´´´powershell python \src\features\substrate_emb_ChamBERTA2.py´´´
VISUALIZATION: ´´´poweshell python .\src\utils\visualize_substrate_embeddings.py´´´
ANALYSIS: ´´´powershell $env:OMP_NUM_THREADS = "1"
python .\scripts\analyze_substrate_embeddings.py´´´
And for the conection between cluster classes and activity:
´´´python .\scripts\analyze_cluster_properties.py´´´
---

### 🚀 Quick Start Guide

#### Step-by-Step: Training a Model

1. **Prepare your data** (if not already done):
   ```bash
   python scripts/fetch_data.py
   python scripts/concatenate_embeddings.py --substrate chemberta3
   ```

2. **Choose your approach:**

   **Option A: Quick Training (with default parameters)**
   ```bash
   python scripts/train_nn.py
   ```
   Uses settings from `configs/neural_network.yml`. Good for testing or if you already have optimized parameters.

   **Option B: Hyperparameter Optimization First (Recommended)**
   ```bash
   # Find best hyperparameters (takes ~45-60 minutes)
   python scripts/tune_nn_optuna.py
   
   # Copy best parameters from output to configs/neural_network.yml
   # Then train with optimized settings:
   python scripts/train_nn.py
   ```
   Optuna automatically tests 50 different configurations to find the best model.

3. **View results:**
   - Training curves: `reports/figures/nn_training_*.png`
   - Metrics JSON: `reports/metrics/nn_metrics_*.json`
   - Best model: `experiments/best_model_*.pth`

#### Making Predictions with Trained Model

```python
import torch
from src.models.nn_model import AttentionMLP, GT_NN, BilinearInteractionNet

# Load model
model = AttentionMLP(input_dim=1792, hidden_dims=[512, 256], num_heads=4)
model.load_state_dict(torch.load('experiments/best_model_chemberta3.pth'))
model.eval()

# Load your concatenated embedding
X_test = np.load('data/concatenated_embeddings/X_chemberta3.npy')

# Make prediction
with torch.no_grad():
    prediction = model(torch.FloatTensor(X_test[0:1]))
    probability = torch.sigmoid(prediction).item()
    print(f"Activity probability: {probability:.2%}")
```

---

### 🤖 Neural Network Training

Train neural networks for GT-substrate activity prediction using concatenated embeddings.

#### Available Model Types

1. **Standard MLP (`model_type: "simple"`)**
   - Flexible feedforward network with configurable layers
   - Examples: `[512, 256]` for 2 layers, `[1024, 512, 256, 128]` for deeper networks
   - Fast and effective baseline

2. **Bilinear Interaction Network (`model_type: "bilinear"`)**
   - Explicitly models protein-substrate interactions via projections
   - Computes Hadamard product and dot product between features
   - Projects embeddings to lower dimensions for efficiency

3. **Attention MLP (`model_type: "attention"`)** - Best Performance ✅
   - Cross-attention mechanism between protein and substrate
   - Multi-head attention captures diverse interaction patterns
   - Bidirectional: protein attends to substrate AND substrate attends to protein
   - Residual connections + LayerNorm for stable training
   - **Best results**: 84.2% C1, 93% C2, 98.2% ROC-AUC (ChemBERTa3, 4 heads)

#### Configuration

Edit `configs/neural_network.yml`:

```yaml
# Model architecture
model_type: "attention"  # "simple", "bilinear", or "attention"
hidden_dims: [512, 256]  # Hidden layer dimensions (flexible - can be any depth)
dropout: 0.4  # Dropout rate (0.3-0.5 recommended)

# Attention-specific parameters (only for model_type: "attention")
num_heads: 4  # Number of attention heads (4 recommended, 8 may overfit)
use_residual: true  # Use residual connections (recommended)

# Data augmentation (optional)
data_augmentation: false  # Enable Gaussian noise during training
noise_std: 0.02  # Standard deviation for noise (if augmentation enabled)

# Training hyperparameters
learning_rate: 0.001
batch_size: 16
epochs: 100
weight_decay: 0.0005  # L2 regularization

# Data
substrate_name: "chemberta3"  # "chemberta2", "chemberta3", or "kpgt"
protein_name: "prott5"

# W&B tracking
wandb_mode: "offline"  # "online" or "offline"
```

#### Running Training

**Basic training:**
```bash
python scripts/train_nn.py
```

**Monitor training:**
- Real-time metrics printed to console
- Training curves auto-generated after completion
- Weights & Biases tracking (if configured)

#### Outputs

- **Model checkpoint**: `experiments/best_model_{substrate_name}.pth` - Saved automatically when validation accuracy improves
- **Training curves**: `reports/figures/nn_training_{substrate_name}_{timestamp}.png` - Loss, accuracy, F1, ROC-AUC plots
- **Metrics JSON**: `reports/metrics/nn_metrics_{substrate_name}_{timestamp}.json` - Complete evaluation results
- **W&B logs**: `wandb/offline-run-*/` (if enabled) - Sync with `wandb sync <run-dir>` for online dashboard

---

### 🎯 Hyperparameter Optimization with Optuna

Automatically find the best hyperparameters using Optuna - a smart optimization framework that learns from previous trials.

#### How to Use

1. **Install Optuna:**
```bash
pip install optuna
```

2. **Run hyperparameter optimization:**
```bash
python scripts/tune_nn_optuna.py
```

**What happens during optimization:**
- Prunes unpromising trials early
- Tracks all metrics: F1-score (primary), accuracy, ROC-AUC, loss
- Generates comprehensive 6-plot visualization dashboard

3. **Review results:**

The script automatically saves:
- `reports/optuna/best_params_{substrate_name}.json` - Best hyperparameters found
- `reports/optuna/optuna_study_{substrate_name}.png` - 6-plot visualization dashboard:
  - Optimization history with rolling average
  - Parameter importance ranking
  - Learning rate vs performance
  - Dropout vs performance  
  - Model architecture comparison
  - Substrate embedding comparison

#### Customizing Optimization

**What Optuna searches over:**
- **Substrate embeddings**: ChemBERTa2 (384D), ChemBERTa3 (768D), KPGT (2304D)
- **Model architectures**: MLP, Attention MLP, Bilinear Interaction Network
- **Network depth**: 1-3 hidden layers
- **Layer widths**: [128, 256, 512, 768]
- **Learning rate**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
- **Regularization**: Dropout (0.2-0.9), Weight decay (1e-5 to 1e-3)
- **Training**: Batch size (8, 16, 32)
- **Attention-specific**: Number of heads (2, 4, 8), Projection dim (64, 128, 256)

**Modify search parameters:**

Edit `scripts/tune_nn_optuna.py` at the bottom (lines ~495-505):
```python
config = {
    'n_trials': 1000,      # Number of configurations to test
    'max_epochs': 60,    # Max epochs per trial (most converge by 30)
    'timeout': None,     # Optional: max time in seconds (e.g., 3600)
}
```

**Change optimization metric:**

By default, Optuna maximizes F1-score. To optimize for a different metric, edit line ~228:
```python
# Current: return val_f1
# Options: return val_accuracy, val_roc_auc, val_f1
```

**Adjust pruning aggressiveness:**

Edit line ~275 in `objective()` function:
```python
# Current: MedianPruner(n_startup_trials=5, n_warmup_steps=10)
# More aggressive (faster): n_warmup_steps=5
# Less aggressive (thorough): n_warmup_steps=15
```
---

### 📈 Experiments
- **Baseline models:** Random classifier, majority class, logistic regression.
- **Neural networks:** Flexible MLP, Bilinear Interaction Network, Attention MLP.
- **Hyperparameter optimization:** Optuna for automated tuning.
- **Future work:** GNN on molecular graphs, multi-modal transformer, AlphaFold2 integration.

#### Baseline models
In order to run the baseline models one has to specify the settings of the experiment in the configs folder in a yaml file e.g. the scikit-learn.yml

##### Create a virtual environment
1. Install uv with 
    ```sh
    pip install uv
    ```
2. Install dependencies with

    ```sh
    uv sync
    ```
    This will create a virtual environment `.venv` under project folder and install all the dependencies listed in the `pyproject.toml` file.
    
3. You can add packages to uv 
  Add package and update files
  ```bash
  uv add package-name
  uv sync
  ```
  Make sure to include the new pyproject.toml and uv.lock file in the commit, otherwise uv sync will not install them for other users.
4. Enter the virtual enviroment 

  ```sh
    source .venv/bin/activate
  ```
##### Login with wandb
 ```sh
    uv add wandb
    uv run wandb login
  ```

##### Run the experiment within the env
 ```sh
    python scripts/run_experiment.py
    ```

### 💡 Results
tba

### References
tba
