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

#### Data Processing and Embedding Generation Pipeline

1. Dataset Preprocessing
    Entry point:
    ```bash
    python run_data_preprocessing.py
    ```
    This step prepares all raw input data required for downstream processing. It includes:
      - Filtering and standardizing substrate SMILES
      - Preparing KPGT-compatible input files
      - Creating FASTA files for protein sequences
      - Merging original data sources
      - Creating the full, cleaned interaction dataset

    Creating the original dataset:

   1. Add a .env file in the project root with:
      ACCESS_DB_PATH=/full/path/to/database.accdb
      ACCESS_DB_PASSWORD=yourpassword

   2. Run the script, the function create_original_dataset will:
      - Check if UGT.csv, Activity.csv, and Substrate.csv already exist in data/
      - If missing, export the tables from the .accdb file and retrieve additional substrate information

   3. Output: CSV files will be saved in the data/ folder, ready for preprocessing and further analysis.

   
**This step only needs to be run once unless the raw data changes.**


2. Dataset Splitting (C1 / C2 / C3)
    Entry point:
    ```bash
    python run_data_split.py
    ```
    This script performs dataset splitting according to enzyme–substrate generalization settings:
     - Training
     - Validation (C1 / C2 / C3 splits)
     - Test (C1 / C2 / C3 splits)
   
    **The splits are stored explicitly in the dataset (via a split column or separate files), allowing downstream steps to operate without recomputing splits.**


3. Embedding Generation and Concatenation
    Entry point:
    ```bash
    python run_generate_embeddings.py
    ```
    This script generates embeddings for proteins and substrates and optionally concatenates them for model training.
     - Protein embeddings 
       - (ProtT5 (1024D))
     - Substrate embeddings
       - ChemBERTa-2 
       - ChemBERTa-3 
       - KPGT
       
    You can configure which embeddings are generated and whether concatenation is performed via function arguments.
    
    **About KPGT:**
    KPGT embeddings are generated using the method described in:
    "A Knowledge-Guided Pre-training Framework for Improving Molecular Representation Learning"

    The implementation is not included directly in this repository.

    Before generating KPGT embeddings, you must clone the official KPGT repository:
    ```bash
    git clone https://github.com/lihan97/KPGT
    ```
    Follow the installation and environment setup instructions provided in the KPGT repository.
    Once installed, the embedding generation script in this project will call the KPGT code as part of the embedding pipeline.
    
    **Add Substrate and Protein Features into the Embeddings (Optional)** 

    This project allows you to append substrate features and protein features to the final concatenated embedding vector. These features are configured via the YAML file `configs/features.yml`.

    1. Enable or Disable All Features Inside `configs/features.yml`:
       ```
       features:
       use_features: true
       ```
       - Set to true → when you run the commands in 3. Generate Concatenated Embeddings, the substrate features and protein features will be automatically computed and appended to the concatenated embedding vector. Final vector becomes:[ protein_embedding || substrate_embedding || selected_features ] 
       - Set to false → the concatenation process will produce embeddings that contain only the original protein and substrate embedding vectors, without adding any features.

    2. Available Features (Reference Lists) 

        The YAML file defines two complete catalogs of all implemented features: 
           - `available_substrate_features:` — all substrate feature types supported by the project 
           - `available_protein_features:` — all protein feature types supported by the project 

        These lists exist only as references. They do not affect the model unless selected. 

    3. Active Features (Used in Model Input) 

       To use any feature during concatenation, you must explicitly list it:
        - Add substrate features to `substrate_features:` 
        - Add protein features to `protein_features:` 

       Only the names listed under these two sections will be computed and appended to the concatenated embedding vector. 

       Feature selection summary: 
            - `available_*_features` = everything you could choose 
            - `*_features` = the items you actually activate by listing them

4. Recommended Usage Order
    
    To run the full pipeline from raw data to embeddings:
 
    ```bash
    python run_data_preprocessing.py
    python run_data_split.py
    python run_generate_embeddings.py
    ```
   
    **Each step can be rerun independently if needed (e.g. regenerating embeddings without reprocessing data)**

#### Clustering GT sequences with MMseqs2
1.  Install MMseqs2

    - Go to the MMseqs2 GitHub releases page:
    
    👉 https://github.com/soedinglab/MMseqs2/releases
    
    - Download mmseqs-win64.zip (Windows)
    
    - Extract it to the tools/ folder inside your project (so you have tools/mmseqs/bin/mmseqs.bat)
    
    - You can either:

        - Use the full path when running it, or 

        - Add tools/mmseqs/bin to your PATH environment variable.

2. Run the clustering:

   - Once MMseqs2 is ready, run the clustering command (adjust filenames if needed, UGT.fasta is created by the preprocessing step):
    ```powershell 
    .\tools\mmseqs\bin\mmseqs.exe easy-cluster data\UGT.fasta data\GT_cluster tmp --min-seq-id 0.7 -c 0.8
    ```
   - min-seq-id 0.5 sets 50% minimum sequence identity
    
   - c 0.8 sets 80% minimum coverage

3. Output files: After running, MMseqs2 will generate several output files:

    - GT_cluster_cluster.tsv → sequence-to-cluster assignments
    
    - GT_cluster_rep_seq.fasta → one representative sequence per cluster
    
    - GT_cluster_all_seqs.fasta → all clustered sequences

4. Report output:
   ```powershell 
     python .\scripts\print_cluster_report.p
    ``` 
    for the report output -> CONCLUSION: 90% redundancy reduction

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

1. **Standard MLP**
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
   - Bidirectional: protein attends to substrate and substrate attends to protein
   - Residual connections + LayerNorm for stable training


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
data_augmentation: true  # Enable Gaussian noise during training
noise_std: 0.02  # Standard deviation for noise (if augmentation enabled)

# Training hyperparameters
learning_rate: 0.001
batch_size: 16
epochs: 100
weight_decay: 0.0005  # L2 regularization

# Oversampling
oversample: true  # Minority class oversampling in training set

# Feature incorporation
use_handcrafted_features: true  # Before using it run: python scripts/extract_features_full_dataset.py

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

**Ensemble Setup:**
- 5 models with different random seeds: [42, 123, 456, 789, 1337]
- Simple averaging of prediction probabilities
- Each model trained independently with data augmentation
- Saves to: `experiments/best_model_chemberta2_aug_seed_{seed}.pth`

**Running the Ensemble:**
```bash
# Train ensemble models (run 5 times with different seeds)
python scripts/train_nn.py --seed 42 --save_path experiments/best_model_chemberta2_aug_seed_42.pth
python scripts/train_nn.py --seed 123 --save_path experiments/best_model_chemberta2_aug_seed_123.pth
python scripts/train_nn.py --seed 456 --save_path experiments/best_model_chemberta2_aug_seed_456.pth
python scripts/train_nn.py --seed 789 --save_path experiments/best_model_chemberta2_aug_seed_789.pth
python scripts/train_nn.py --seed 1337 --save_path experiments/best_model_chemberta2_aug_seed_1337.pth

# Evaluate ensemble
python scripts/ensemble_predict.py --models experiments/best_model_chemberta2_aug_seed_*.pth

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
- **Network depth**: 1-4 hidden layers
- **Layer widths**: [128, 256, 512, 1024]
- **Learning rate and scheduler**: [1e-6, 1e-5, 1e-4, 1e-3, 1e-2], StepLR (step size, gamma) or ReduceOnPlateau 
- **Regularization**: Dropout (0.2-0.9), Weight decay (1e-6 to 1e-2)
- **Optimizer**: Adam, AdamW, SGD eith momentum
- **Training**: Batch size (16, 32, 64)
- **Activation function**: ReLu, GeLu, Tanh, Sigmoid, Leaky ReLu
- **Attention-specific**: Number of heads (2, 4, 8), Residual conncections (True, False)
- **Bilinear model specific**: (64, 128, 256)

**Modify search parameters:**

Edit `scripts/tune_nn_optuna.py` at the bottom (lines ~495-505):
```python
config = {
    'n_trials': 150,      # Number of configurations to test
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
- **Neural networks:** Simple MLP, Deep MLP, Bilinear Interaction Network.
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

#### Best Model Performance

Our final production model uses the below written parameters from Optuna study and with Gaussian noise (std=0.02) , label smoothing (0.05), oversampling on minority class (Random sampler on inactive protein-substrate pairs) and incorporates hand-crafted features for improving performance.  

**Performance on Test Sets:**

| Test Set | F1 Score (±SE) | Accuracy (±SE) | ROC-AUC (±SE) | MCC (±SE) |
|----------|----------------|----------------|---------------|----------|
| **C1** | 0.909 ± 0.021 | 0.865 ± 0.031 | 0.932 ± 0.022 | 0.648 ± 0.077 |
| **C2** | 0.732 ± 0.026 | 0.812 ± 0.017 | 0.881 ± 0.015 | 0.595 ± 0.035 |
| **C3** | 0.727 ± 0.116 | 0.739 ± 0.091 | 0.983 ± 0.023 | 0.586 ± 0.118 | 

**Key Findings:**
- **Improved C2/C3 balance** - Enhanced regularization stack reduced overfitting, C3 accuracy improved significantly
- Comprehensive regularization (gradient clipping, mixup, stochastic depth) helps model generalize to unseen proteins/substrates
- **C3 ROC-AUC (0.983)** shows excellent ranking ability despite moderate accuracy
- Handcrafted features combined with aggressive regularization provide robust performance
- Standard errors computed via bootstrap (1000 samples) for reliable uncertainty quantification

**Training Configuration:**
```yaml
# Best configuration from experiments
substrate_name: chemberta3
model_type: attention (num_heads = 2 , use_residual = True)
hidden_dims: [1024, 512] (n_layers = 2)

# Core hyperparameters
dropout: 0.4
weight_decay: 0.00011761899025546045
learning_rate: 0.0001
batch_size: 32
optimizer: AdamW
scheduler: StepLR (step_size = 7, gamma = 0.8)
activation: tanh

# Regularization stack
data_augmentation: true
noise_std: 0.02              # Gaussian noise augmentation
label_smoothing: 0.05        # Label smoothing for calibration
grad_clip: 1.0               # Gradient clipping (prevents explosions)
mixup_alpha: 0.2             # Mixup augmentation (smooth boundaries)
stochastic_depth: 0.1        # DropPath layer regularization

# Data techniques
oversample: true
use_handcrafted_features: true
```

#### 🛡️ Regularization Techniques

Our model uses a comprehensive 10-layer regularization stack to prevent overfitting and improve generalization:

**1. Gradient Clipping** (`grad_clip: 1.0`)
- Prevents gradient explosions during backpropagation
- Stabilizes training, especially for attention mechanisms
- Clips gradients to max norm of 1.0

**2. Mixup Augmentation** (`mixup_alpha: 0.2`)
- Creates virtual training samples by interpolating between pairs
- Forces model to learn smooth decision boundaries
- Particularly effective for reducing C2/C3 overfitting gap

**3. Stochastic Depth (DropPath)** (`stochastic_depth: 0.1`)
- Randomly drops entire layers during training (10% probability)
- Reduces co-adaptation between consecutive layers
- Creates implicit ensemble effect

**4. Gaussian Noise** (`noise_std: 0.02`)
- Adds random noise to input embeddings during training
- Makes model robust to small perturbations
- Prevents memorization of exact embedding values

**5. Label Smoothing** (`label_smoothing: 0.05`)
- Smooths binary labels: 0→0.05, 1→0.95
- Prevents overconfident predictions
- Improves probability calibration

**6. Dropout** (`dropout: 0.4`)
- Randomly drops 40% of neurons per layer
- Classic regularization for redundant representations

**7. L2 Regularization** (`weight_decay: 0.00011...`)
- Penalizes large weights in loss function
- Encourages distributed, smaller weights

**8. Batch Normalization** (built-in)
- Normalizes activations within mini-batches
- Provides mild regularization via batch statistics

**9. Learning Rate Scheduling** (StepLR)
- Reduces LR by 20% every 7 epochs
- Allows refined updates as training progresses

**10. Early Stopping** (patience=10)
- Stops when validation loss plateaus
- Prevents training past optimal generalization point

**What We Tested:**
1. ✅ **Gradient Clipping** - Critical for training stability
2. ✅ **Mixup Augmentation** - Significantly improved C2/C3 generalization
3. ✅ **Stochastic Depth** - Reduced layer co-adaptation, better regularization
4. ✅ **Gaussian Noise** - Improved robustness to input perturbations
5. ✅ **Label Smoothing** - Better probability calibration
6. ✅ **Oversampling** - Addressed class imbalance effectively
7. ✅ **Handcrafted Features** - Improved overall performance
8. ❌ **Ensemble Method** - Simple averaging did not outperform single models

### References

ESP data:
Kroll, A., Ranjan, S., Engqvist, M.K.M., Lercher, M.J. "A general model to predict small molecule substrates of enzymes based on machine and deep learning." Nature Communications, 14, 2787 (2023). https://doi.org/10.1038/s41467-023-38347-2

EZSpecificity data:
Cui, H., Su, Y., Dean, T.J., Yu, T., Zhang, Z., Peng, J., Shukla, D., Zhao, H. "Enzyme specificity prediction using cross-attention graph neural networks." Nature, 647, 639–647 (2025). https://doi.org/10.1038/s41586-025-09697-2

GT-Predict evaluation:
Yang, M., Fehl, C., Lees, K.V., Lim, E-H., Offen, W.A., Davies, G.J., Bowles, D.J. "Functional and informatics analysis enables glycosyltransferase activity prediction." Nature Chemical Biology, 14, 1109–1117 (2018). https://doi.org/10.1038/s41589-018-0154-9

