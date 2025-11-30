import numpy as np
import torch
import pickle

# ChemBERTa2
cb2 = np.load('embeddings/substrate_embeddings_chemberta2.npy')
print('ChemBERTa2:', cb2.shape)

# ChemBERTa3
cb3 = torch.load('embeddings/ChemBERTa3_substrate_embeddings.pt', map_location='cpu')
if isinstance(cb3, dict):
    print('ChemBERTa3 keys:', cb3.keys())
    if 'embeddings' in cb3:
        print('ChemBERTa3:', cb3['embeddings'].shape)
    elif 'substrate_embeddings' in cb3:
        print('ChemBERTa3:', cb3['substrate_embeddings'].shape)
    else:
        for k, v in cb3.items():
            if hasattr(v, 'shape'):
                print(f'ChemBERTa3[{k}]:', v.shape)
else:
    print('ChemBERTa3:', cb3.shape)

# KPGT
kpgt = np.load('embeddings/kpgt_substrate_embeddings.npz')
print('KPGT:', kpgt['fps'].shape)

# Protein
prot = np.load('data/Protein_Embeddings/protein_embeddings_prott5.npy')
print('ProtT5:', prot.shape)
