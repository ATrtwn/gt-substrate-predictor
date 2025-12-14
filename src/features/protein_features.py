from Bio.SeqUtils.ProtParam import ProteinAnalysis


def compute_protein_features(sequence: str, selected: list[str]):
    # Handle empty / invalid sequence
    if not isinstance(sequence, str) or len(sequence.strip()) == 0:
        return {name: None for name in selected}

    # Clean sequence
    sequence = sequence.replace("\r", "").replace("\n", "")
    sequence = sequence.strip().upper()

    if len(sequence) == 0:
        return {name: None for name in selected}

    p = ProteinAnalysis(sequence)
    length = len(sequence)

    # --- Amino acid composition-based groups ---
    aa_counts = p.count_amino_acids()  # dict like {"A": 10, "C": 3, ...}
    total = float(length) if length > 0 else 1.0  # avoid division by zero

    # Groups (very simple, global, composition-based)
    hydrophobic = set("AILMVFWY")
    positive = set("KRH")
    negative = set("DE")
    polar = set("STNQYC")

    def _frac(residues: set[str]) -> float:
        return sum(aa_counts.get(a, 0) for a in residues) / total

    frac_hydrophobic = _frac(hydrophobic)
    frac_positive = _frac(positive)
    frac_negative = _frac(negative)
    frac_polar = _frac(polar)

    # --- Base features from BioPython ---
    feature_map = {
        "length": length,
        "aromaticity": p.aromaticity(),
        "instability_index": p.instability_index(),
        "isoelectric_point": p.isoelectric_point(),
        "gravy": p.gravy(),  # hydrophobicity (global)
        "frac_hydrophobic": frac_hydrophobic,  # fraction of hydrophobic residues
        "frac_positive": frac_positive,        # fraction of positively charged residues
        "frac_negative": frac_negative,        # fraction of negatively charged residues
        "frac_polar": frac_polar,              # fraction of polar residues
    }

    return {name: feature_map.get(name, None) for name in selected}
