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

Generate CSV files from the Access database

1. Create a `.env` file in the project root with:

    - ACCESS_DB_PATH=/full/path/to/database.accdb
    - ACCESS_DB_PASSWORD=yourpassword

2. Run the script:

   python scripts/fetch_data.py
    
   The script will:
   - Check if UGT.csv, Activity.csv, and Substrate.csv already exist in data/
   - If missing, export the tables from the .accdb file and fetch additional substrate info

3. Result: CSV files will be saved in the data/ folder, ready for preprocessing and analysis.

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
