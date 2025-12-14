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

#### 2. Add Substrate and Protein Features into the Embeddings (Optional) 
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

#### 3. Generate Concatenated Embeddings

After generating protein embeddings (ProtT5) and substrate embeddings (ChemBERTa2, ChemBERTa3, KPGT), concatenate them for ML model training. If substrate/protein features are enabled in `configs/features.yml`, these features will be appended to the concatenated embedding as well.:

**Generate all substrate embedding types:**
```bash
python scripts/concatenate_embeddings.py --substrate all
```

**Generate specific substrate type only:**
```bash
# ChemBERTa2 (384D) + ProtT5 (1024D) = 1408D (+ features if enabled)
python scripts/concatenate_embeddings.py --substrate chemberta2

# ChemBERTa3 (768D) + ProtT5 (1024D) = 1792D (+ features if enabled)
python scripts/concatenate_embeddings.py --substrate chemberta3

# KPGT (2304D) + ProtT5 (1024D) = 3328D (+ features if enabled)
python scripts/concatenate_embeddings.py --substrate kpgt
```
#### 4. Run the clustering:
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
- Adds selected features if `use_features: true`
- Generates 2251 valid concatenated pairs (100% coverage)

#### 5. Report output:
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

### 🤖 Neural Network Training

Train neural networks for GT-substrate activity prediction using concatenated embeddings.

#### Available Model Types

1. **Simple MLP (`model_type: "simple"`)**
   - 3-layer feedforward network
   - Fast and effective baseline

2. **Deep MLP (`model_type: "deep"`)**
   - 5-layer deep network with residual-style connections

3. **Bilinear Interaction Network (`model_type: "bilinear"`)**
   - Explicitly models protein-substrate interactions
   - Projects embeddings to lower dimensions

4. **Attention MLP (`model_type: "attention"`)**
   - Cross-attention mechanism between protein and substrate
   - Multi-head attention captures diverse interaction patterns

#### Configuration

Edit `configs/neural_network.yml`:

```yaml
# Model architecture
model_type: "attention"  # "simple", "deep", "bilinear", or "attention"
hidden_dims: [512, 256]  # Hidden layer dimensions
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

```bash
# Train with current config
python scripts/train_nn.py
```

The script will:
- Automatically normalize embeddings (StandardScaler)
- Create stratified train/val/test splits
- Apply early stopping (patience=10)
- Save best model checkpoint
- Generate training curves (loss, accuracy, F1, ROC-AUC)
- Save metrics to JSON file
- Log to Weights & Biases

#### Outputs

- **Model checkpoint**: `experiments/best_model_{substrate_name}.pth`
- **Training curves**: `reports/figures/nn_training_{substrate_name}_{id}.png`
- **Metrics JSON**: `reports/metrics/nn_metrics_{substrate_name}_{id}.json`
- **W&B logs**: `wandb/offline-run-*/` (sync with `wandb sync <run-dir>`)
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
tba

### References
tba
