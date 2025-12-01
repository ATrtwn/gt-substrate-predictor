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

### 📈 Experiments
- **Baseline models:** Random classifier, majority class, logistic regression.
- **Initial models:** FNN on concatenated embeddings.
- **Advanced models:** GNN on molecular graphs, multi-modal transformer.
- **Structural integration:** AlphaFold2-derived binding pocket features.

### 💡 Results
tba

### References
tba
