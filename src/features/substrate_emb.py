# ChemBERTa / KPGT embeddings
import argparse
from multiprocessing import freeze_support

import sys
import os

# Add project root to sys.path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print("Project root:", project_root)
sys.path.append(project_root)

from third_party.kpgt.scripts.preprocess_downstream_dataset import preprocess_dataset
from third_party.kpgt.scripts.extract_features import extract_features

class Args:
    def __init__(self, data_path, dataset, path_length):
        self.data_path = data_path
        self.dataset = dataset
        self.path_length = path_length
        self.n_jobs = 32
        self.config = "base"
        self.model_path = f"{project_root}/models/pretrained/base/base.pth"

# Usage
args = Args(data_path="your/data/path", dataset="your_dataset", path_length=5)

def get_kpgt_embedding(args: Args):
    print("Starting KPGT embedding extraction...")
    #preprocess_dataset(args= args)
    extract_features(args= args)
 
def main ():
    data_path = f"{project_root}/data"
    dataset = "Substrate"
    path_length = 5
    args = Args(data_path = data_path, dataset = dataset, path_length = path_length)
    get_kpgt_embedding(args = args)

if __name__ == '__main__':
    freeze_support()  # Optional, but recommended for frozen applications
    main()