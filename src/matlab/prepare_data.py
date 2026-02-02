import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors
from rdkit.Chem import rdFreeSASA
from rdkit.Chem import AllChem
from accfg import AccFG
from rdkit import RDLogger

# Disable all RDKit warnings
RDLogger.DisableLog('rdApp.*')

def label_flavonoid_positions(mol, a_ring, b_ring, c_ring):
    pos = {}

    a_set = set(a_ring)
    b_set = set(b_ring)
    c_set = set(c_ring)

    # ---- find shared atoms between A and C ----
    shared = list(a_set & c_set)
    if len(shared) != 2:
        raise ValueError("Expected exactly two shared A/C ring atoms")

    s1, s2 = shared

    # ---- find heterocyclic O in C-ring ----
    c_ring_o = None
    for i in c_ring:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == "O":
            c_ring_o = i
            break
    if c_ring_o is None:
        raise ValueError("No O atom in C-ring")

    # ---- determine which shared atom is position 9 ----
    def is_neighbor(a, b):
        return any(n.GetIdx() == b for n in mol.GetAtomWithIdx(a).GetNeighbors())

    if is_neighbor(s1, c_ring_o):
        pos9, pos10 = s1, s2
    elif is_neighbor(s2, c_ring_o):
        pos9, pos10 = s2, s1
    else:
        raise ValueError("Could not identify position 9")

    pos[c_ring_o] = 1
    pos[pos9] = 9
    pos[pos10] = 10

    # ---- A-ring walk: 8 → 7 → 6 → 5 ----
    # start from 9
    current = pos9
    position_number = 8
    # for next atom in a ring take neighbor (e.g. for 9 neighbors are 10 and 8)
    walk = True
    while (walk):
        # find neighbors in the ring that are not already labeled (e.g. for 9 one neighbor, 10, already has label, so go to 8 next)
        next = [
            n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in a_set and n.GetIdx() not in pos.keys()
        ]

        if not next:
            # when both neighbors have labels stop (e.g. when we reach 5 then both neighbors, 6 and 10, have labels)
            break

        # take the first neighbor as next
        next_atom = next[0]
        next_label = position_number
        pos[next_atom] = next_label

        position_number-=1
        current = next_atom

    # ---- C-ring walk: O → 2 → 3 → 4 → 5 ----
    # start from O
    current = c_ring_o
    position_number = 2
    # for next atom in a ring take neighbor
    walk = True
    while (walk):
        # find neighbors in the ring that are not already labeled
        next = [
            n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in c_set and n.GetIdx() not in pos.keys()
        ]

        if not next:
            # when both neighbors have labels stop
            break

        # take the first neighbor as next
        next_atom = next[0]
        next_label = position_number
        pos[next_atom] = next_label

        position_number += 1
        current = next_atom

    # ---- B-ring walk: 1' → 2' → 3' → 4' → 5' → 6'  ----
    # Identify 1' as the atom in B-ring bonded to C-ring
    for atom_idx in b_ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        if any(nbr.GetIdx() in c_set for nbr in atom.GetNeighbors()):
            one_prime = atom_idx
            break

    # Walk around B-ring to assign positions
    # Use neighbor traversal, avoid going back to already visited atom
    visited = []
    current = one_prime
    while len(visited) < len(b_ring):
        atom = mol.GetAtomWithIdx(current)
        # neighbors in B-ring only
        nbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if
                nbr.GetIdx() in b_set and nbr.GetIdx() not in visited]
        visited.append(current)
        if not nbrs:
            break
        current = nbrs[0]

    # mark positions 4' and 5' as 13 and 14
    idx_1_prime = visited[0]
    idx_2_prime = visited[1]
    idx_3_prime = visited[2]
    idx_4_prime = visited[3]
    idx_5_prime = visited[4]
    idx_6_prime = visited[5]
    # para to 1' is 4'
    pos[idx_4_prime] = 13
    # two possibilities for 14
    pos[idx_3_prime] = 14
    pos[idx_5_prime] = 14
    #
    pos[idx_1_prime] = 11
    pos[idx_2_prime] = 12
    pos[idx_6_prime] = 16

    return pos

def label_coumarin_positions(mol, a_ring, c_ring):
    pos = {}

    a_set = set(a_ring)
    c_set = set(c_ring)

    # ---- find shared atoms between A and C ----
    shared = list(a_set & c_set)
    if len(shared) != 2:
        raise ValueError("Expected exactly two shared A/C ring atoms")

    s1, s2 = shared

    # ---- find heterocyclic O in C-ring ----
    c_ring_o = None
    for i in c_ring:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetSymbol() == "O":
            c_ring_o = i
            break
    if c_ring_o is None:
        raise ValueError("No O atom in C-ring")

    # ---- determine which shared atom is position 9 ----
    def is_neighbor(a, b):
        return any(n.GetIdx() == b for n in mol.GetAtomWithIdx(a).GetNeighbors())

    if is_neighbor(s1, c_ring_o):
        pos9, pos10 = s1, s2
    elif is_neighbor(s2, c_ring_o):
        pos9, pos10 = s2, s1
    else:
        raise ValueError("Could not identify position 9")

    pos[c_ring_o] = 1
    pos[pos9] = 9
    pos[pos10] = 10

    # ---- A-ring walk: 8 → 7 → 6 → 5 ----
    # start from 9
    current = pos9
    position_number = 8
    # for next atom in a ring take neighbor (e.g. for 9 neighbors are 10 and 8)
    walk = True
    while (walk):
        # find neighbors in the ring that are not already labeled (e.g. for 9 one neighbor, 10, already has label, so go to 8 next)
        next = [
            n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in a_set and n.GetIdx() not in pos.keys()
        ]

        if not next:
            # when both neighbors have labels stop (e.g. when we reach 5 then both neighbors, 6 and 10, have labels)
            break

        # take the first neighbor as next
        next_atom = next[0]
        next_label = position_number
        pos[next_atom] = next_label

        position_number -= 1
        current = next_atom

    # ---- C-ring walk: O → 2 → 3 → 4 → 5 ----
    # start from O
    current = c_ring_o
    position_number = 2
    # for next atom in a ring take neighbor
    walk = True
    while (walk):
        # find neighbors in the ring that are not already labeled
        next = [
            n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in c_set and n.GetIdx() not in pos.keys()
        ]

        if not next:
            # when both neighbors have labels stop
            break

        # take the first neighbor as next
        next_atom = next[0]
        next_label = position_number
        pos[next_atom] = next_label

        position_number += 1
        current = next_atom

    return pos

def label_cytokinins_positions(mol, a_ring, b_ring):
    pos = {}

    a_set = set(a_ring)
    b_set = set(b_ring)

    ring_atoms = set(a_ring) | set(b_ring)
    chain = set(i for i in range(mol.GetNumAtoms()) if i not in ring_atoms)

    # ---- find atoms bond to rest of the molecule for A and B ----
    for atom_idx in a_ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        if any(nbr.GetIdx() in chain for nbr in atom.GetNeighbors()):
            bond_position_a = atom_idx
            break

    # ---- find shared atoms between A and C ----
    shared = list(a_set & b_set)
    if len(shared) != 2:
        raise ValueError("Expected exactly two shared A/C ring atoms")

    s1, s2 = shared

    # Find N3: the N in the A-ring bonded to either s1 or s2
    def is_neighbor(a, b):
        return any(n.GetIdx() == b for n in mol.GetAtomWithIdx(a).GetNeighbors())

    n3 = None
    for n_atom_idx in a_ring:
        atom = mol.GetAtomWithIdx(n_atom_idx)
        if atom.GetSymbol() == "N" and (is_neighbor(n_atom_idx, s1) or is_neighbor(n_atom_idx, s2)):
            n3 = n_atom_idx
            break

    if n3 is None:
        raise ValueError("Could not find N3 in A-ring")

    # ---- determine which shared atom is position 4 ----
    # position 4 is neighbor to N3
    if is_neighbor(s1, n3):
        pos4, pos5 = s1, s2
    elif is_neighbor(s2, n3):
        pos4, pos5 = s2, s1
    else:
        raise ValueError("Could not identify position 5")

    pos[n3] = 3
    pos[pos5] = 5
    pos[pos4] = 4

    # ---- A-ring walk: 4 → 3 → 2 → 1 → 6  ----
    a_sequence = [4, 3, 2, 1, 6]
    # start from 4
    current = pos4
    position_number = 1
    # for next atom in a ring take neighbor (e.g. for 9 neighbors are 10 and 8)
    walk = True
    while (walk):
        # find neighbors in the ring that are not already labeled (e.g. for 9 one neighbor, 10, already has label, so go to 8 next)
        next = [
            n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in a_set and n.GetIdx() not in pos.keys()
        ]

        if not next:
            # when both neighbors have labels stop (e.g. when we reach 5 then both neighbors, 6 and 10, have labels)
            break

        # take the first neighbor as next
        next_atom = next[0]
        next_label = a_sequence[position_number]
        pos[next_atom] = next_label

        position_number += 1
        current = next_atom

    # ---- B-ring walk: 4 → 9 → 8 → 7 ----
    b_sequence = [4, 9, 8, 7]
    # start from 4
    current = pos4
    position_number = 1
    # for next atom in a ring take neighbor
    walk = True
    while (walk):
        # find neighbors in the ring that are not already labeled
        next = [
            n.GetIdx() for n in mol.GetAtomWithIdx(current).GetNeighbors()
            if n.GetIdx() in b_set and n.GetIdx() not in pos.keys()
        ]

        if not next:
            # when both neighbors have labels stop
            break

        # take the first neighbor as next
        next_atom = next[0]
        next_label = b_sequence[position_number]
        pos[next_atom] = next_label

        position_number += 1
        current = next_atom

    return pos

def label_cinnamate_positions(mol, aromatic_ring):
    pos = {}

    ring_set = set(aromatic_ring)
    chain = set(i for i in range(mol.GetNumAtoms()) if i not in ring_set)

    # ---- find atom bond to rest of the molecule ----
    for atom_idx in aromatic_ring:
        atom = mol.GetAtomWithIdx(atom_idx)
        if any(nbr.GetIdx() in chain for nbr in atom.GetNeighbors()):
            bond_position = atom_idx
            break

    # walk through ring and assign positions for Cn2-OH, Cn3-OH and Cn4-OH
    visited = []
    current = bond_position
    while len(visited) < len(aromatic_ring):
        atom = mol.GetAtomWithIdx(current)
        # neighbors in B-ring only
        nbrs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if
                nbr.GetIdx() in ring_set and nbr.GetIdx() not in visited]
        visited.append(current)
        if not nbrs:
            break
        current = nbrs[0]

    # mark positions 3', 4' and 5'
    idx_1_prime = visited[0]
    idx_2_prime = visited[1]
    idx_3_prime = visited[2]
    idx_4_prime = visited[3]
    idx_5_prime = visited[4]
    idx_6_prime = visited[5]
    #
    pos[idx_4_prime] = 4
    pos[idx_3_prime] = 3
    pos[idx_5_prime] = 5
    pos[idx_1_prime] = 1
    pos[idx_2_prime] = 2
    pos[idx_6_prime] = 6

    return pos

def has_OH_substituent(atom):
    """
    atom = ring carbon
    returns True if atom has an -OH substituent
    """
    if atom.GetSymbol() != "C":
        return False

    for nbr in atom.GetNeighbors():
        if nbr.GetSymbol() == "O":
            # check if that oxygen has a hydrogen
            if any(n.GetSymbol() == "H" for n in nbr.GetNeighbors()):
                return True
    return False

def has_substituent(atom, ring_atoms):
    """
    Returns True if the atom has a neighbor not in its own ring
    """
    return any(n.GetIdx() not in ring_atoms for n in atom.GetNeighbors())

def get_scaffolds_info(smiles, verbose=False):
    ########## RDKit
    # Create RDKit molecule from SMILES and add explicit hydrogens
    mol_rdkit = Chem.MolFromSmiles(smiles)
    rings = mol_rdkit.GetRingInfo().AtomRings()
    if verbose:
        print("SMILES (canonical):", Chem.MolToSmiles(mol_rdkit))
        print("Num atoms:", mol_rdkit.GetNumAtoms())
        print("Num rings:", mol_rdkit.GetRingInfo().NumRings())
        print("Ring atom sets:", rings)

    # === detect flavonoid core broadly ===
    features = {"F3_OH": 0.0, "F5_OH": 0.0, "F6_OH": 0.0, "F7_OH": 0.0, "F13_OH": 0.0, "F14_OH": 0.0}
    try:
        f_patterns = ["c(~*)c(-c2ccccc2)occc(O)ccc", "O=c1occc2ccccc12"]
        for p in f_patterns:
            FLAVONOID_CORE = Chem.MolFromSmarts(p)
            has_f_core = mol_rdkit.HasSubstructMatch(FLAVONOID_CORE)
            if has_f_core:
                f_pattern = p
                if verbose:
                    print(f"Flavonoid pattern: {f_pattern}")
                break
        if verbose:
            print(f"Flavonoid core: ", has_f_core)
        if has_f_core:
            mol_rdkit = Chem.AddHs(mol_rdkit)
            core_match = mol_rdkit.GetSubstructMatch(FLAVONOID_CORE)
            core_atoms = set(core_match)
            if verbose:
                print(f"Flavonoid core matches: {core_match}")

            # ---- identify C-ring: heterocycle with O + carbonyl ----
            c_ring = None
            for ring in rings:
                ring_set = set(ring)
                if not ring_set & core_atoms:
                    continue

                has_oxygen = any(
                    mol_rdkit.GetAtomWithIdx(i).GetSymbol() == "O"
                    for i in ring
                )

                has_carbonyl = any(
                    mol_rdkit.GetAtomWithIdx(i).GetSymbol() == "C" and
                    any(
                        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                        mol_rdkit.GetAtomWithIdx(bond.GetOtherAtomIdx(i)).GetSymbol() == "O"
                        for bond in mol_rdkit.GetAtomWithIdx(i).GetBonds()
                    )
                    for i in ring
                )

                if has_oxygen and has_carbonyl:
                    c_ring = ring
                    break

            if c_ring is None:
                raise ValueError(f"Could not identify C-ring for {Chem.MolToSmiles(mol_rdkit)}")
            else:
                if verbose:
                    print(f"C-ring: {c_ring}")

            # ---- identify A-ring: fused aromatic ring ----
            a_ring = None
            c_ring_set = set(c_ring)

            for ring in rings:
                if ring == c_ring:
                    continue
                if len(set(ring) & c_ring_set) >= 2:
                    a_ring = ring
                    break

            if a_ring is None:
                raise ValueError(f"Could not identify A-ring for {Chem.MolToSmiles(mol_rdkit)}")
            else:
                if verbose:
                    print(f"A-ring: {a_ring}")

            # ---- identify B-ring: phenyl attached to C-ring ----
            b_ring = None
            for ring in rings:
                if ring in (c_ring, a_ring):
                    continue
                ring_set = set(ring)
                # must be aromatic
                if not all(mol_rdkit.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    continue
                # must be bonded to C-ring
                for i in ring:
                    atom = mol_rdkit.GetAtomWithIdx(i)
                    if any(nbr.GetIdx() in c_ring_set for nbr in atom.GetNeighbors()):
                        b_ring = ring
                        break
                if b_ring is not None:
                    break

            if b_ring is None:
                raise ValueError(f"Could not identify B-ring for {Chem.MolToSmiles(mol_rdkit)}")
            else:
                if verbose:
                    print(f"B-ring: {b_ring}")

            if verbose:
                # ---- inspect atoms ----
                print("Check for A Ring")
                for atom_idx in a_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

                print("Check for B Ring")
                for atom_idx in b_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

                print("Check for C Ring")
                for atom_idx in c_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

            # ---- find positions ----
            pos = label_flavonoid_positions(mol_rdkit, a_ring, b_ring, c_ring)

            # ---- assign features ----
            for atom_idx in c_ring:
                if pos[atom_idx] == 3:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["F3_OH"] = 1.0

            for atom_idx in a_ring:
                if pos[atom_idx] == 5:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["F5_OH"] = 1.0
                if pos[atom_idx] == 6:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["F6_OH"] = 1.0
                if pos[atom_idx] == 7:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["F7_OH"] = 1.0

            for atom_idx in b_ring:
                if pos[atom_idx] == 13:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["F13_OH"] = 1.0
                if pos[atom_idx] == 14:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["F14_OH"] = 1.0

            if verbose:
                print("Flavonoid OH positional features:", features)
    except Exception as e:
        if verbose:
            print(f"Flavonoid detection failed → treating as non-flavonoid: {e}")

    # === detect coumarin core broadly ===
    features.update({"Cm6-OH": 0.0, "Cm7-OH": 0.0})
    try:
        cm_patterns = ["O=c1ccc2ccc(O)cc2o1",
                       "O=c1cc(O)c2ccccc2o1",
                       "O=c1ccc2occc2c1"]
        for p in cm_patterns:
            COUMARIN = Chem.MolFromSmarts(p)
            has_cm_core = mol_rdkit.HasSubstructMatch(COUMARIN)
            if has_cm_core:
                cm_pattern = p
                if verbose:
                    print(f"Coumarin pattern: {cm_pattern}")
                break
        if verbose:
            print(f"Coumarin core: ", has_cm_core)
        if has_cm_core:
            mol_rdkit = Chem.AddHs(mol_rdkit)
            core_match = mol_rdkit.GetSubstructMatch(COUMARIN)
            core_atoms = set(core_match)
            if verbose:
                print(f"Coumarin core matches: {core_match}")

            rings = mol_rdkit.GetRingInfo().AtomRings()

            # ---- identify C-ring (pyrone) ----
            c_ring = None
            for ring in rings:
                ring_set = set(ring)
                if not ring_set & core_atoms:
                    continue
                has_oxygen = any(
                    mol_rdkit.GetAtomWithIdx(i).GetSymbol() == "O"
                    for i in ring
                )
                has_carbonyl = any(
                    mol_rdkit.GetAtomWithIdx(i).GetSymbol() == "C" and
                    any(
                        bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and
                        mol_rdkit.GetAtomWithIdx(bond.GetOtherAtomIdx(i)).GetSymbol() == "O"
                        for bond in mol_rdkit.GetAtomWithIdx(i).GetBonds()
                    )
                    for i in ring
                )
                if has_oxygen and has_carbonyl:
                    c_ring = ring
                    break

            if c_ring is None:
                raise ValueError(f"Could not identify coumarin C-ring for {Chem.MolToSmiles(mol_rdkit)}")
            else:
                if verbose:
                    print(f"Coumarin C-ring: {c_ring}")

            # ---- identify A-ring: fused aromatic ring ----
            a_ring = None
            c_ring_set = set(c_ring)
            for ring in rings:
                if ring == c_ring:
                    continue
                if len(set(ring) & c_ring_set) >= 2:
                    a_ring = ring
                    break

            if a_ring is None:
                raise ValueError(f"Could not identify coumarin A-ring for {Chem.MolToSmiles(mol_rdkit)}")
            else:
                if verbose:
                    print(f"Coumarin A-ring: {a_ring}")

            if verbose:
                # ---- inspect atoms ----
                print("Check for A Ring")
                for atom_idx in a_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

                print("Check for C Ring")
                for atom_idx in c_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

            # ---- find positions ----
            pos = label_coumarin_positions(mol_rdkit, a_ring, c_ring)

            # ---- assign features ----
            for atom_idx in a_ring:
                if pos[atom_idx] == 6:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["Cm6-OH"] = 1.0
                if pos[atom_idx] == 7:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["Cm7-OH"] = 1.0

            if verbose:
                print("Coumarin OH positional features:", features)
    except Exception as e:
        if verbose:
            print(f"Coumarin detection failed → treating as non-coumarin: {e}")

    # === detect Cytokinins core broadly ===
    features.update({"Ck3-N": 0.0, "Ck7-N": 0.0, "Ck-OH": 0.0})
    try:
        ck_patterns = ["n1cnc2ncnc12",
                       "CNC2=NC=NC3=C2NC=N3",
                       "C(O)NC2=NC=NC3=C2NC=N3",
                       "CC(C)=CNC2=NC=NC3=C2NC=N3",
                       "c1ccccc1CNC2=NC=NC3=C2NC=N3",]
        for p in ck_patterns:
            CYTOKININS = Chem.MolFromSmarts(p)
            has_ck_core = mol_rdkit.HasSubstructMatch(CYTOKININS)
            if has_ck_core:
                ck_pattern = p
                if verbose:
                    print(f"Cytokinin pattern: {ck_pattern}")
                break
        if verbose:
            print(f"Cytokinin core: ", has_ck_core)
        if has_ck_core:
            mol_rdkit = Chem.AddHs(mol_rdkit)
            core_match = mol_rdkit.GetSubstructMatch(CYTOKININS)
            core_atoms = set(core_match)
            if verbose:
                print(f"Cytokinin core matches: {core_match}")

            rings = mol_rdkit.GetRingInfo().AtomRings()

            # ---- identify aromatic rings ----
            aromatic_rings = []
            for ring in rings:
                if all(mol_rdkit.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_rings.append(ring)

            if len(aromatic_rings) < 2:
                raise ValueError(f"Could not identify both cytokinin rings for {Chem.MolToSmiles(mol_rdkit)}")

            # ---- identify Six-membered pyrimidine ring and Five-membered imidazole ring ----
            a_ring = None  # pyrimidine (6-membered)
            b_ring = None  # imidazole (5-membered)

            for ring in rings:
                if len(ring) == 6:
                    # pyrimidine: 2 nitrogens typically
                    n_count = sum(
                        mol_rdkit.GetAtomWithIdx(i).GetSymbol() == "N"
                        for i in ring
                    )
                    if n_count >= 2:
                        a_ring = ring

                elif len(ring) == 5:
                    # imidazole: 2 nitrogens
                    n_count = sum(
                        mol_rdkit.GetAtomWithIdx(i).GetSymbol() == "N"
                        for i in ring
                    )
                    if n_count >= 2:
                        b_ring = ring

            if a_ring is None or b_ring is None:
                raise ValueError(f"Could not identify cytokinin rings properly for {Chem.MolToSmiles(mol_rdkit)}")

            if verbose:
                print(f"A-ring: {a_ring}")
                print(f"B-ring: {b_ring}")

            ring_atoms = set(a_ring) | set(b_ring)
            chain = set(i for i in range(mol_rdkit.GetNumAtoms()) if i not in ring_atoms)

            if verbose:
                # ---- inspect atoms ----
                print("Check for A Ring")
                for atom_idx in a_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

                print("Check for B Ring")
                for atom_idx in b_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

            # ---- find positions ----
            pos = label_cytokinins_positions(mol_rdkit, a_ring, b_ring)

            # ---- assign features ----
            for atom_idx in a_ring:
                if pos[atom_idx] == 3:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_substituent(atom, a_ring):
                        features["Ck3-N"] = 1.0

            for atom_idx in b_ring:
                if pos[atom_idx] == 7:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_substituent(atom, b_ring):
                        features["Ck7-N"] = 1.0

            for atom_idx in chain:
                atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                if has_OH_substituent(atom):
                    features["Ck-OH"] = 1.0

            if verbose:
                print("Cytokinin OH positional features:", features)

    except Exception as e:
        if verbose:
            print(f"Cytokinin detection failed → treating as non-cytokinin: {e}")

    # === detect cinnamate core broadly ===
    features.update({"Cn2-OH": 0.0, "Cn3-OH": 0.0, "Cn4-OH": 0.0})
    try:
        cn_patterns = ["O=C(O)/C=C/c1ccccc1"]
        for p in cn_patterns:
            CINNAMATE = Chem.MolFromSmarts(p)
            has_cn_core = mol_rdkit.HasSubstructMatch(CINNAMATE)
            if has_ck_core:
                cn_pattern = p
                if verbose:
                    print(f"Cinnamate pattern: {cn_pattern}")
                break
        if verbose:
            print(f"Cinnamate core: ", has_cn_core)
        if has_cn_core:
            mol_rdkit = Chem.AddHs(mol_rdkit)
            core_match = mol_rdkit.GetSubstructMatch(CINNAMATE)
            core_atoms = set(core_match)
            if verbose:
                print(f"Cinnamate core matches: {core_match}")

            rings = mol_rdkit.GetRingInfo().AtomRings()

            # ---- identify the aromatic ring ----
            aromatic_ring = None
            for ring in rings:
                if any(i in core_atoms for i in ring) and all(mol_rdkit.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
                    aromatic_ring = ring
                    break

            if aromatic_ring is None:
                raise ValueError(f"Could not identify cinnamate aromatic ring for {Chem.MolToSmiles(mol_rdkit)}")
            else:
                if verbose:
                    print(f"Aromatic ring: {aromatic_ring}")

            if verbose:
                # ---- inspect atoms ----
                print("Check for Aromatic Ring")
                for atom_idx in aromatic_ring:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    print(atom_idx, atom.GetSymbol(), atom.GetTotalNumHs(),
                          atom.GetHybridization(), atom.GetDegree())

            # ---- find positions ----
            pos = label_cinnamate_positions(mol_rdkit, aromatic_ring)

            # ---- assign features ----
            for atom_idx in aromatic_ring:
                if pos[atom_idx] == 3:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["Cn2-OH"] = 1.0
                if pos[atom_idx] == 4:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["Cn3-OH"] = 1.0
                if pos[atom_idx] == 5:
                    atom = mol_rdkit.GetAtomWithIdx(atom_idx)
                    if has_OH_substituent(atom):
                        features["Cn4-OH"] = 1.0

            if verbose:
                print("Cinnamate OH positional features:", features)

    except Exception as e:
        if verbose:
            print(f"Cinnamate detection failed → treating as non-cinnamate: {e}")

    return features

def find_relaxed_core(df, min_substrates=10, min_enzymes=5, max_iter=20):
    core = df.copy()

    for _ in range(max_iter):
        changed = False

        enz_counts = core.groupby("UGT_ID")["substrate"].nunique()
        keep_enz = enz_counts[enz_counts >= min_substrates].index
        if len(keep_enz) < core["UGT_ID"].nunique():
            core = core[core["UGT_ID"].isin(keep_enz)]
            changed = True

        sub_counts = core.groupby("substrate")["UGT_ID"].nunique()
        keep_sub = sub_counts[sub_counts >= min_enzymes].index
        if len(keep_sub) < core["substrate"].nunique():
            core = core[core["substrate"].isin(keep_sub)]
            changed = True

        if not changed:
            break

    enzymes = core["UGT_ID"].unique()
    substrates = sorted(core["substrate"].unique())

    print(
        f"Dense core: "
        f"{len(enzymes)} enzymes × {len(substrates)} substrates"
    )
    return core

def infer_family(scaffolds_features):
    if any(scaffolds_features.get(k, 0) > 0 for k in ["F3_OH","F5_OH","F6_OH"]):
        return 1  # flavonoid-like
    if any(scaffolds_features.get(k, 0) > 0 for k in ["Cm6-OH","Cm7-OH"]):
        return 2  # coumarin-like
    if any(scaffolds_features.get(k, 0) > 0 for k in ["Ck3-N", "Ck7-N", "Ck-OH"]):
        return 3  # cytokinin-like
    if any(scaffolds_features.get(k, 0) > 0 for k in ["Cn2-OH","Cn3-OH","Cn4-OH"]):
        return 4  # cinnamate-like
    return 0

def get_substrate_features(substrates_df, verbose=False):
    rows = []

    for _, row in substrates_df.drop_duplicates("substrate").iterrows():
        features = {}

        smiles = row["SMILES_isomeric_1"]
        mol_rdkit = Chem.MolFromSmiles(smiles)
        if mol_rdkit is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        ########## RDKit
        # Create RDKit molecule from SMILES and add explicit hydrogens
        if verbose:
            print(f"=== Name: {row['substrate']} ===")

        # add Hs
        mol_rdkit = Chem.AddHs(mol_rdkit)

        # Generate a 3D conformer and optimize geometry (required for volume/SASA)
        try:
            status = AllChem.EmbedMolecule(mol_rdkit, randomSeed=42)
        except RuntimeError as e:
            if verbose:
                print(f"[WARN] Embedding failed for {smiles}: {e}")
            status = -1

        if status != 0:
            if verbose:
                print(f"[WARN] Embedding failed for {smiles}, skipping 3D features")
            volume = np.nan
            area = np.nan

        else:
            AllChem.MMFFOptimizeMolecule(mol_rdkit)
            # source: https://periodictable.com/Properties/A/VanDerWaalsRadius.v.html
            vdw_radii = {
                'H': 1.2, 'C': 1.7, 'N': 1.55, 'O': 1.52, 'F': 1.47,
                'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98
            }
            radii = {atom.GetIdx(): vdw_radii.get(atom.GetSymbol(), 1.5) for atom in mol_rdkit.GetAtoms()}

            # volume
            def sphere_volume(r):
                return (4.0 / 3.0) * np.pi * r ** 3

            volume = 0.0
            for atom in mol_rdkit.GetAtoms():
                r = vdw_radii.get(atom.GetSymbol(), 1.5)
                volume += sphere_volume(r)

            if verbose:
                print("Approx. vdW volume (Å³):", volume)

            area = rdFreeSASA.CalcSASA(mol_rdkit, radii=radii)

        # Physicochemical
        features.update({
            "LogP": Descriptors.MolLogP(mol_rdkit),
            "AccessibleArea": area,
            "Volume": volume
        })

        ########## AccFG
        # Count common groups
        afg = AccFG(lite=True)
        fgs, _ = afg.run(
            smiles,
            show_atoms=True,
            show_graph=True,
            canonical=False
        )

        features.update({
            "Num_OH": int(len(fgs.get("hydroxy", [])) + len(fgs.get("primary hydroxyl", [])) +
                          len(fgs.get("secondary hydroxyl", [])) + len(fgs.get("tertiary hydroxyl", [])) +
                          len(fgs.get("phenol", []))),
            "COOH": int(len(fgs.get("carboxylic acid", [])) + len(fgs.get("Carboxylic acid", [])))
        })

        if verbose:
            print(f"Features = {features}")

        scaffolds_features = get_scaffolds_info(smiles, verbose=verbose)

        row = {
            "Name": row["substrate"],
            "Family": infer_family(scaffolds_features),
            "LogP": features["LogP"],
            "AccessibleArea": features["AccessibleArea"],
            "Volume": features["Volume"],
            "COOH": features["COOH"],  # pKa or proxy
            "Num_OH": features["Num_OH"],

            # scaffold features (zeros if unknown)
            "F3-OH": scaffolds_features.get("F3_OH", 0.0),
            "F5-OH": scaffolds_features.get("F5_OH", 0.0),
            "F6-OH": scaffolds_features.get("F6_OH", 0.0),
            "F7-OH": scaffolds_features.get("F7_OH", 0.0),
            "F13-OH": scaffolds_features.get("F13_OH", 0.0),
            "F14-OH": scaffolds_features.get("F14_OH", 0.0),

            "Cm6-OH": scaffolds_features.get("Cm6-OH", 0.0),
            "Cm7-OH": scaffolds_features.get("Cm7-OH", 0.0),

            "Ck3-N": scaffolds_features.get("Ck3-N", 0.0),
            "Ck7-N": scaffolds_features.get("Ck7-N", 0.0),
            "Ck-OH": scaffolds_features.get("Ck-OH", 0.0),

            "Cn2-OH": scaffolds_features.get("Cn2-OH", 0.0),
            "Cn3-OH": scaffolds_features.get("Cn3-OH", 0.0),
            "Cn4-OH": scaffolds_features.get("Cn4-OH", 0.0),
        }

        rows.append(row)

    feats_df = pd.DataFrame(rows)

    return feats_df

def build_interaction_matrix_enzyme(all_pairs_df, out_file, random=False):

    df = all_pairs_df.copy()

    enzymes = df["UGT_ID"].unique()
    substrates = sorted(df["substrate"].unique())

    # split enzymes
    if random:
        # random split
        print("Using random split")
        np.random.seed(42)
        test_frac = 0.2

        test_enzymes = np.random.choice(enzymes, size=int(len(enzymes) * test_frac), replace=False)
        train_enzymes = np.setdiff1d(enzymes, test_enzymes)

        train_df = df[df["UGT_ID"].isin(train_enzymes)].copy()
        test_df = df[df["UGT_ID"].isin(test_enzymes)].copy()
    else:
        # split with min sequence similarity between train and test
        print("Using cluster split")
        test_df = df[df["split"].isin(["C1_test", "C2_test", "C3_test"])].copy()
        test_enzymes = set(test_df["UGT_ID"].unique())

        train_df = df[~df["split"].isin(["C1_test", "C2_test", "C3_test"])].copy()
        train_enzymes = set(train_df["UGT_ID"].unique())

    # build matrix from train only
    # Header 1: enzyme names
    enzymes_str = [str(e) for e in train_enzymes]
    header_enzyme_names = ["", ""] + enzymes_str

    # Header 2: enzyme family (all glycotransferases)
    header_enzyme_families = ["", ""] + ['G'] * len(train_enzymes)

    X = pd.DataFrame(2, index=substrates, columns=sorted(train_enzymes))

    train_enzyme_interactions = df[df["UGT_ID"].isin(train_enzymes)].copy()

    for _, row in train_enzyme_interactions.iterrows():
        X.loc[row["substrate"], row["UGT_ID"]] = row["is_active"]

    # minimal required metadata
    meta = pd.DataFrame({
        "ID": range(1, len(substrates) + 1),
        "Name": substrates
    })

    full = pd.concat([meta, X.reset_index(drop=True)], axis=1)

    with open(out_file, 'w') as f:
        # Header row 1: enzyme names
        f.write('\t'.join(header_enzyme_names) + '\n')
        # Header row 2: enzyme family
        f.write('\t'.join(header_enzyme_families) + '\n')
        # Data
        full.to_csv(f, sep='\t', index=False, header=False)
        # full.to_csv(out_file, sep="\t", index=False)

    print(f"Saved Enzyme interation matrix to {out_file}")

    # return test interactions for evaluation
    return test_df, train_df, meta

def build_interaction_matrix_acceptor(all_pairs_df, out_file, random=False):

    df = all_pairs_df.copy()

    enzymes = df["UGT_ID"].unique()
    substrates = sorted(df["substrate"].unique())

    # split substrates
    if random:
        # random split
        print("Using random split")
        np.random.seed(42)
        test_frac = 0.2

        test_substrates = np.random.choice(substrates, size=int(len(substrates) * test_frac), replace=False)
        train_substrates = np.setdiff1d(substrates, test_substrates)

        train_df = df[df["substrate"].isin(train_substrates)].copy()
        test_df = df[df["substrate"].isin(test_substrates)].copy()
    else:
        # split with min sequence similarity between train and test
        print("Using cluster split")
        test_df = df[df["split"].isin(["C1_test", "C2_test", "C3_test"])].copy()
        test_substrates = set(test_df["substrate"].unique())

        train_df = df[~df["split"].isin(["C1_test", "C2_test", "C3_test"])].copy()
        train_substrates = set(train_df["substrate"].unique())

    # build matrix from train only
    X = pd.DataFrame(2, index=sorted(train_substrates), columns=sorted(enzymes))

    train_substrates_interactions = df[df["UGT_ID"].isin(train_substrates)].copy()

    for _, row in train_substrates_interactions.iterrows():
        X.loc[row["substrate"], row["UGT_ID"]] = row["is_active"]

    # fill the substrate features
    feats_df = get_substrate_features(df[['substrate', 'molecule', 'SMILES_isomeric_1']], verbose=False)
    feats_train = feats_df[feats_df['Name'].isin(train_substrates)].copy()
    feats_train.insert(0, "ID", range(1, len(feats_train) + 1))

    full = pd.concat([feats_train, X.reset_index(drop=True)], axis=1)
    full.to_csv(out_file, sep="\t", index=False)

    print(f"Saved Acceptor interation matrix to {out_file}")

    # return test interactions for evaluation
    return test_df, train_df, feats_df