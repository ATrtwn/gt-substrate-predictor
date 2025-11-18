
def filter_dataset(df, min_seq_len=50, min_mol_weight=100):
    """Apply basic filtering criteria to remove incomplete or extreme data points."""
    pass

def mmseqs_clustering(fasta_path, output_dir, identity_threshold=0.9, coverage=0.8):
    """Run MMseqs2 clustering and redundancy reduction"""
    pass

def stratified_split_by_entities(df, protein_col="UGT_ID", substrate_col="substrate", random_state=42):
    """
    Perform stratified split according to generalization classes (C1, C2, C3).

    Splits data such that:
        - C1: both gt and substrate seen in training
        - C2: one unseen (either gt or substrate)
        - C3: both unseen
    """
    pass

def balance_activity_classes(df, label_col="activity", method="undersample"):
    """
    Balance active vs. inactive samples in the training dataset.
    """
    pass

def preprocess_pipeline(raw_data_dir, processed_dir):
    """Main preprocessing pipeline combining clustering, standardization, and splitting."""
    pass

