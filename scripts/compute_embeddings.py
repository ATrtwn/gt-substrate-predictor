import pandas as pd
import torch
from gt_substrate_predictor.features.protein_emb import *

df = pd.read_csv("../data/UGT.csv")
seqs = df["prot_seq"].tolist()
names = df['UGT_trivial_name'].tolist()

tokenizer, model, device = load_prott5_model()

embeddings = compute_prott5_embeddings(seqs, tokenizer, model, device, batch_size=64)

torch.save(embeddings, "embeddings.pt")