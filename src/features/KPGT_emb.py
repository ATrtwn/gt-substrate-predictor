# KPGT embeddings
from multiprocessing import freeze_support
import sys
import os
from pathlib import Path

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

# Add project root to sys.path so imports work
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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
        self.model_path = f"{project_root}/third_party/kpgt/src/model/pretrained/base/base.pth"

# Usage
# args = Args(data_path="your/data/path", dataset="your_dataset", path_length=5)

def generate_KPGT_emb(verbose=False):
    freeze_support()  # Optional, but recommended for frozen applications
    dataset = "Substrate"
    path_length = 5
    args = Args(data_path=data_dir, dataset=dataset, path_length=path_length)
    if verbose:
        print("    Starting KPGT embedding extraction...")
    preprocess_dataset(args=args)
    extract_features(args=args)
    if verbose:
        print("    Finished KPGT embedding extraction...")