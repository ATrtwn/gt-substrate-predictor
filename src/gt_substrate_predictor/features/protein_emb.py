import torch
import re
from transformers import T5Tokenizer, T5EncoderModel

def preprocess_protein_sequence(seq: str) -> str:
    """
    Replace ambiguous amino acids (U, Z, O, B) with X
    and insert spaces between each residue.
    """
    seq = re.sub(r"[UZOB]", "X", seq)
    return " ".join(list(seq))

def load_prott5_model(device=None):
    """
    Load ProtT5 tokenizer and encoder model.
    Returns tokenizer, model, device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = T5Tokenizer.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc",
        do_lower_case=False,
    )

    model = T5EncoderModel.from_pretrained(
        "Rostlab/prot_t5_xl_half_uniref50-enc"
    ).to(device)

    if device.type == "cpu":
        model.to(torch.float32)

    return tokenizer, model, device
 
def compute_prott5_embeddings(sequences: list[str], tokenizer: T5Tokenizer, model: T5EncoderModel, device, batch_size=8):
    """
    Compute per-protein embeddings (mean-pooled)
    using ProtT5. Returns a torch.Tensor of shape (N, 1024).
    """
    processed = [preprocess_protein_sequence(s) for s in sequences]
    all_embeddings = []
    model.eval()

    for i in range(0, len(processed), batch_size):
        batch = processed[i : i + batch_size]
        ids = tokenizer(batch, add_special_tokens=True, padding=True)

        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state  # (B, L, 1024)

        # mean pooling per sequence
        for j in range(hidden.size(0)):
            valid_len = attention_mask[j].sum().item()  # number of tokens incl. special
            # remove <cls> and <sep>
            residue_embeddings = hidden[j, 1 : valid_len - 1]
            emb = residue_embeddings.mean(dim=0)  # (1024,)
            all_embeddings.append(emb.cpu())

    return torch.stack(all_embeddings)