import math 
import yaml 
from pathlib import Path 
from src.features.substrate_features import compute_substrate_features 
from src.features.protein_features import compute_protein_features 
def load_feature_config(): 
    config_path = Path(__file__).parents[2] / "configs" / "features.yml" 
    with open(config_path, "r") as f: 
        raw_cfg = yaml.safe_load(f) 
    return raw_cfg.get("features", {}) 

def _safe_float(val): 
    # Convert feature value to float; None -> nan.
    if val is None: 
        return float("nan") 
    return float(val) 

def compute_all_features(smiles: str, protein_seq: str): 
    cfg = load_feature_config() 
    
    if not cfg.get("use_features", False): 
        return [] 
    
    final_vec = [] 

    # Substrate features 
    sub_keys = cfg.get("substrate_features", []) 
    sub_vals = compute_substrate_features(smiles, sub_keys) 
    final_vec.extend(_safe_float(sub_vals[k]) for k in sub_keys) 
    
    # Protein features 
    prot_keys = cfg.get("protein_features", []) 
    prot_vals = compute_protein_features(protein_seq, prot_keys) 
    final_vec.extend(_safe_float(prot_vals[k]) for k in prot_keys) 
    
    return final_vec