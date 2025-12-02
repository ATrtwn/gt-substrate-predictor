# ChemBERTa / KPGT embeddings
import argparse
from multiprocessing import freeze_support
import sys
import os

# Add project root and KPGT to sys.path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
kpgt_root = os.path.join(project_root, "third_party", "kpgt")
print("Project root:", project_root)
print("KPGT root:", kpgt_root)
sys.path.insert(0, kpgt_root)  # Add KPGT first for its internal imports
sys.path.append(project_root)

from third_party.kpgt.scripts.preprocess_downstream_dataset import preprocess_dataset
from third_party.kpgt.scripts.extract_features import extract_features

class Args:
    def __init__(self, data_path, dataset, path_length, n_jobs=4):
        self.data_path = data_path
        self.dataset = dataset
        self.path_length = path_length
        self.n_jobs = n_jobs  # Reduced for Windows
        self.config = "base"
        self.model_path = f"{project_root}/third_party/models/pretrained/base/base.pth"

def get_kpgt_embedding(args: Args):
    print("\n=== Step 1: Preprocessing dataset ===")
    print(f"Data path: {args.data_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Path length: {args.path_length}")
    preprocess_dataset(args=args)
    
    print("\n=== Step 2: Extracting features ===")
    print(f"Model path: {args.model_path}")
    extract_features(args=args)
    print("\n=== KPGT embedding extraction complete! ===")
 
def main():
    data_path = f"{project_root}/data"
    dataset = "Substrate"
    path_length = 5
    n_jobs = 4  # Use fewer workers on Windows
    
    args = Args(data_path=data_path, dataset=dataset, path_length=path_length, n_jobs=n_jobs)
    
    # Verify files exist
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model not found at {args.model_path}")
        return
    
    substrate_csv = f"{data_path}/{dataset}/{dataset}.csv"
    if not os.path.exists(substrate_csv):
        print(f"ERROR: Substrate CSV not found at {substrate_csv}")
        print("Please run scripts/prepare_kpgt_data.py first")
        return
    
    get_kpgt_embedding(args=args)

if __name__ == '__main__':
    freeze_support()
    main()