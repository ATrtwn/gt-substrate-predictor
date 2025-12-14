from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.Lipinski import NumHDonors, NumHAcceptors


def _count_smarts(mol, smarts: str) -> int:
    # count matches of a SMARTS pattern.
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return 0
    return len(mol.GetSubstructMatches(patt))


def compute_substrate_features(smiles: str, selected: list[str]):
    # Basic checks
    if not isinstance(smiles, str) or smiles.strip() == "":
        return {name: None for name in selected}

    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return {name: None for name in selected}

    if mol is None:
        return {name: None for name in selected}

    # Basic counts
    atoms = list(mol.GetAtoms())
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 0:
        return {name: None for name in selected}

    # Element counts
    num_c = sum(1 for a in atoms if a.GetSymbol() == "C")
    # hydrogen: implicit + explicit
    num_h = sum(a.GetTotalNumHs() for a in atoms)
    num_n = sum(1 for a in atoms if a.GetSymbol() == "N")
    num_o = sum(1 for a in atoms if a.GetSymbol() == "O")
    num_s = sum(1 for a in atoms if a.GetSymbol() == "S")
    num_p = sum(1 for a in atoms if a.GetSymbol() == "P")
    num_f = sum(1 for a in atoms if a.GetSymbol() == "F")
    num_cl = sum(1 for a in atoms if a.GetSymbol() == "Cl")

    # Hetero atoms (non C, non H)
    num_hetero_atoms = sum(
        1 for a in atoms if a.GetSymbol() not in ("C", "H")
    )

    # Hybridization counts (for carbon atoms)
    from rdkit.Chem.rdchem import HybridizationType

    num_sp3_c = sum(
        1
        for a in atoms
        if a.GetSymbol() == "C" and a.GetHybridization() == HybridizationType.SP3
    )
    num_sp2_c = sum(
        1
        for a in atoms
        if a.GetSymbol() == "C" and a.GetHybridization() == HybridizationType.SP2
    )
    num_sp_c = sum(
        1
        for a in atoms
        if a.GetSymbol() == "C" and a.GetHybridization() == HybridizationType.SP
    )

    # Aromatic atoms by element
    num_aromatic_c = sum(
        1 for a in atoms if a.GetIsAromatic() and a.GetSymbol() == "C"
    )
    num_aromatic_n = sum(
        1 for a in atoms if a.GetIsAromatic() and a.GetSymbol() == "N"
    )
    num_aromatic_o = sum(
        1 for a in atoms if a.GetIsAromatic() and a.GetSymbol() == "O"
    )

    # Topological + physchem descriptors
    molwt = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)

    h_donors = NumHDonors(mol)
    h_acceptors = NumHAcceptors(mol)
    h_bond_total = h_donors + h_acceptors

    num_rings = rdMolDescriptors.CalcNumRings(mol)
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
    num_saturated_rings = rdMolDescriptors.CalcNumSaturatedRings(mol)
    num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)

    # Flexibility ratio = rotatable bonds / total atoms
    flexibility_ratio = (
        float(num_rotatable_bonds) / float(n_atoms) if n_atoms > 0 else 0.0
    )

    # Formal charge
    formal_charge = sum(a.GetFormalCharge() for a in atoms)

    # Radical electrons (unpaired electrons)
    num_radical_electrons = sum(
        a.GetNumRadicalElectrons() for a in atoms
    )

    # Valence electrons (manual method)
    VALENCE_TABLE = {
        "H": 1,
        "C": 4,
        "N": 5,
        "O": 6,
        "F": 7,
        "P": 5,
        "S": 6,
        "Cl": 7,
    }

    num_valence_electrons = sum(
        VALENCE_TABLE.get(a.GetSymbol(), 0) for a in atoms
    )

    # Bertz complexity
    bertz_ct = Descriptors.BertzCT(mol)

    # Functional groups via SMARTS (approximate)
    # All hydroxyl groups (OH)
    num_hydroxyl_groups = _count_smarts(mol, "[OX2H]")

    # Carboxyl groups: -C(=O)OH — approximate
    num_carboxyl_groups = _count_smarts(mol, "[CX3](=O)[OX2H1]")

    # Primary amine: -NH2 (non-amide, approximate)
    num_primary_amines = _count_smarts(
        mol, "[NX3;H2;!$(NC=O)]"
    )

    # Aromatic primary amine: -NH2 attached to aromatic carbon
    num_aromatic_primary_amines = _count_smarts(
        mol, "[NX3;H2][c]"
    )

    # Secondary amine: -NH- (non-amide, approximate)
    num_secondary_amines = _count_smarts(
        mol, "[NX3;H1;!$(NC=O)]"
    )

    # Thiol: -SH
    num_thiol_groups = _count_smarts(mol, "[SX2H]")

    # Phenol groups: OH directly on aromatic ring
    num_phenol_groups = _count_smarts(mol, "[OX2H][c]")

    # Aromatic OH groups
    num_aromatic_oh = num_phenol_groups

    # Aliphatic OH groups
    num_aliphatic_oh = max(
        num_hydroxyl_groups - num_aromatic_oh, 0
    )

    # Ratio aromatic/aliphatic OH
    if num_aliphatic_oh > 0:
        ratio_aromatic_to_aliphatic_oh = (
            float(num_aromatic_oh) / float(num_aliphatic_oh)
        )
    else:
        ratio_aromatic_to_aliphatic_oh = 0.0

    # Total GT-reactive functional groups
    num_gt_reactive_groups = (
        num_hydroxyl_groups
        + num_primary_amines
        + num_carboxyl_groups
        + num_thiol_groups
    )

    # Benzene & pyridine rings
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    num_benzene_rings = 0
    num_pyridine_rings = 0

    for ring in atom_rings:
        if len(ring) != 6:
            continue
        ring_atoms = [atoms[i] for i in ring]
        if not all(a.GetIsAromatic() for a in ring_atoms):
            continue

        symbols = [a.GetSymbol() for a in ring_atoms]

        if all(s == "C" for s in symbols):
            num_benzene_rings += 1

        if any(s == "N" for s in symbols):
            num_pyridine_rings += 1

    # Phosphate groups
    num_phosphate_groups = (
        _count_smarts(mol, "[PX4](=O)(O)(O)")
        + _count_smarts(mol, "P(=O)(O)(O)")
    )

    # Ratios
    ratio_aromatic_rings = (
        float(num_aromatic_rings) / float(num_rings)
        if num_rings > 0
        else 0.0
    )
    ratio_heteroatoms = (
        float(num_hetero_atoms) / float(n_atoms)
        if n_atoms > 0
        else 0.0
    )

    # Put everything in one big dict
    feature_map = {
        # basic physchem
        "molwt": molwt,
        "logp": logp,
        "tpsa": tpsa,
        # H-bond related
        "h_donors": h_donors,
        "h_acceptors": h_acceptors,
        "h_bond_total": h_bond_total,
        # rings
        "num_rings": num_rings,
        "num_aromatic_rings": num_aromatic_rings,
        "num_saturated_rings": num_saturated_rings,
        # hetero & rotatable
        "num_hetero_atoms": num_hetero_atoms,
        "num_rotatable_bonds": num_rotatable_bonds,
        "flexibility_ratio": flexibility_ratio,
        # functional groups
        "num_hydroxyl_groups": num_hydroxyl_groups,
        "num_carboxyl_groups": num_carboxyl_groups,
        "num_primary_amines": num_primary_amines,
        "num_aromatic_primary_amines": num_aromatic_primary_amines,
        "num_secondary_amines": num_secondary_amines,
        "num_thiol_groups": num_thiol_groups,
        "num_gt_reactive_groups": num_gt_reactive_groups,
        "num_phenol_groups": num_phenol_groups,
        "num_benzene_rings": num_benzene_rings,
        "num_pyridine_rings": num_pyridine_rings,
        # electrons / charge / complexity
        "formal_charge": formal_charge,
        "num_radical_electrons": num_radical_electrons,
        "num_valence_electrons": num_valence_electrons,
        "bertz_ct": bertz_ct,
        # OH variants
        "num_aromatic_oh": num_aromatic_oh,
        "num_aliphatic_oh": num_aliphatic_oh,
        "ratio_aromatic_to_aliphatic_oh": ratio_aromatic_to_aliphatic_oh,
        # element counts
        "num_c": num_c,
        "num_h": num_h,
        "num_n": num_n,
        "num_o": num_o,
        "num_s": num_s,
        "num_p": num_p,
        "num_f": num_f,
        "num_cl": num_cl,
        # aromatic atoms by element
        "num_aromatic_c": num_aromatic_c,
        "num_aromatic_n": num_aromatic_n,
        "num_aromatic_o": num_aromatic_o,
        # hybridization
        "num_sp3_c": num_sp3_c,
        "num_sp2_c": num_sp2_c,
        "num_sp_c": num_sp_c,
        # ratios
        "ratio_aromatic_rings": ratio_aromatic_rings,
        "ratio_heteroatoms": ratio_heteroatoms,
    }

    # convert outputs to float
    def _to_float(x):
        """
        Convert descriptor outputs to float.
        If conversion fails (e.g. None, numpy types),
        return 0.0 to avoid NaN in ML models.
        """
        try:
            return float(x)
        except Exception:
            return 0.0

    # Return only selected features
    return {
        name: _to_float(feature_map.get(name, 0.0))
        for name in selected
    }
