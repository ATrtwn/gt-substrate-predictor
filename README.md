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
├──.venv # contains the enviroment - has to be created
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

#### Create a virtual environment
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


Generate CSV files from the Access database

1. Create a `.env` file in the project root with:

    - ACCESS_DB_PATH=/full/path/to/database.accdb
    - ACCESS_DB_PASSWORD=yourpassword

2. Run the script(check requirements before):

   python scripts/fetch_data.py
    
   The script will:
   - Check if UGT.csv, Activity.csv, and Substrate.csv already exist in data/
   - If missing, export the tables from the .accdb file and fetch additional substrate info

3. Result: CSV files will be saved in the data/ folder, ready for preprocessing and analysis.

🧬 Clustering GT sequences with MMseqs2

This section explains how to reproduce the clustering of GT sequences using MMseqs2. 

1. Create a FASTA file from the CSV: First, make sure your .csv file. Then run the same Python code we used to generate the FASTA file. This will create a file called UGT.fasta in your project directory.

2. Install MMseqs2
-Go to the MMseqs2 GitHub releases page:
👉 https://github.com/soedinglab/MMseqs2/releases
-Download mmseqs-win64.zip
-Extract it to the tools/ folder inside your project (so you have tools/mmseqs/bin/mmseqs.bat)
-You can either:
  -Use the full path when running it, or
  -Add tools/mmseqs/bin to your PATH environment variable.

3. Run the clustering:
-Once MMseqs2 is ready, run the clustering command (adjust filenames if needed):
tools\mmseqs\bin\mmseqs.exe easy-cluster UGT.fasta GT_cluster tmp --min-seq-id 0.7 -c 0.7
  -min-seq-id 0.7 sets 70% minimum sequence identity (agreed on the meeting)
  -c 0.7 sets 70% minimum coverage (share ≥70% of their length)

4. Output files: After running, MMseqs2 will generate several output files:
  -GT_cluster_cluster.tsv → sequence-to-cluster assignments
  -GT_cluster_rep_seq.fasta → one representative sequence per cluster
  -GT_cluster_all_seqs.fasta → all clustered sequences

5. Report output:
  -run python .\scripts\print_cluster_report.p for the report output -> CONCLUSION: dataset is diverse enough, no need for omiting the sequences

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
