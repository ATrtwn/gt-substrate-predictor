import pandas as pd
import random
import networkx as nx
from collections import defaultdict
from src.utils.visualization import (plot_split_graph,
                                     plot_graph_connectivity,
                                     plot_connectivity_separate,
                                     plot_degree_distributions)

def stratified_split_by_entities(df, protein_col="UGT_ID", substrate_col="substrate",label_col = "is_active", random_state=42, plot=False):
    """
    Perform stratified split according to generalization classes (C1, C2, C3).

    Splits data such that:
        - C1: both gt and substrate seen in training
        - C2: one unseen (either gt or substrate)
        - C3: both unseen
    """
    graph = create_graph(df,
                         protein_col=protein_col,
                         substrate_col=substrate_col,
                         label_col=label_col)


    # split and re-balance if evaluation sets are too small
    # choose parameters
    train_frac = 0.7
    test_frac = 0.15
    reserve_frac = 0.12
    edge_split = split_graph(graph,
                             train_frac=train_frac,
                             test_frac=test_frac,
                             reserve_frac=reserve_frac,
                             random_state=random_state)

    # balance sets, by moving single components to unseen
    train = edge_split['train']
    C1_val = edge_split["C1_val"]
    C2_val = edge_split["C2_val"]
    C3_val = edge_split['C3_val']
    C1_test = edge_split["C1_test"]
    C2_test = edge_split["C2_test"]
    C3_test = edge_split['C3_test']

    min_test_c3 = len(df)*test_frac*0.1
    while((len(C3_test) < 10) and (len(C1_test) > 1)):
        # all val edges in one list
        val = C1_val + C2_val + C3_val
        # choose an edge from c1
        random.seed(random_state)
        edge_to_move = random.choice(C1_test)
        u, v, l, t = edge_to_move
        # choose which component to move from seen to unseen
        unseen_component = u # protein as they are underrepresented in data
        # remove now unseen components from train and val sets
        train_removed = [e for e in train if unseen_component in e]
        train = [e for e in train if unseen_component not in e]
        val_removed = [e for e in val if unseen_component in e]
        val = [e for e in val if unseen_component not in e]
        # re-assign pairs in evaluation sets
        test = C1_test + C2_test + C3_test + train_removed + val_removed
        (C1_val, C2_val, C3_val,
         C1_test, C2_test, C3_test) = create_c_sets(train, val, test)

    min_val_c3 = len(df) * test_frac * 0.1
    while ((len(C3_val) < 10) and (len(C1_val) > 1)):
        # choose an edge from c1
        random.seed(random_state)
        edge_to_move = random.choice(C1_val)
        u, v, l, t = edge_to_move
        # choose which component to move from seen to unseen
        unseen_component = u  # protein as they are underrepresented in data
        # remove now unseen components from train and val sets
        train_removed = [e for e in train if unseen_component in e]
        train = [e for e in train if unseen_component not in e]
        # re-assign pairs in validation sets
        val = C1_val + C2_val + C3_val + train_removed
        test = C1_test + C2_test + C3_test
        (C1_val, C2_val, C3_val,
         C1_test, C2_test, C3_test) = create_c_sets(train, val, test)

    ### plot the split graph
    if plot:
        # all val edges in one list
        val = C1_val + C2_val + C3_val
        seen_nodes = set()
        for u, v, l, t in train:
            seen_nodes.add(u)
            seen_nodes.add(v)
        for u, v, l, t in val:
            seen_nodes.add(u)
            seen_nodes.add(v)
        # make plots
        plot_split_graph(graph, train, val, C1_test, C2_test, C3_test, seen_nodes)
        plot_graph_connectivity(graph)
        plot_connectivity_separate(graph)
        plot_degree_distributions(graph)
    ###

    # perform split on df based on graph split
    df_split = df.copy()
    df_split['split'] = ''

    def assign_split(row, edge_list, split_name):
        if row['split'] == '':
            if (row['UGT_ID'], row['substrate'], row['is_active']) in edge_list:
                return split_name
        return row['split']

    c1_val_pairs = [(u, v, l) for (u, v, l, t) in C1_val]
    c2_val_pairs = [(u, v, l) for (u, v, l, t) in C2_val]
    c3_val_pairs = [(u, v, l) for (u, v, l, t) in C3_val]
    c1_test_pairs = [(u, v, l) for (u, v, l, t) in C1_test]
    c2_test_pairs = [(u, v, l) for (u, v, l, t) in C2_test]
    c3_test_pairs = [(u, v, l) for (u, v, l, t) in C3_test]
    train_pairs = [(u, v, l) for (u, v, l, t) in train]

    for pairs, split_name in [
        (train_pairs, 'train'),
        (c1_val_pairs, 'C1_val'),
        (c2_val_pairs, 'C2_val'),
        (c3_val_pairs, 'C3_val'),
        (c1_test_pairs, 'C1_test'),
        (c2_test_pairs, 'C2_test'),
        (c3_test_pairs, 'C3_test')
    ]:
        df_split['split'] = df_split.apply(lambda row: assign_split(row, pairs, split_name), axis=1)
    df_split.sort_values('split', inplace=True)

    return df_split

def create_graph(df, protein_col="UGT_ID", substrate_col="substrate", label_col="is_active"):
    """
        Create graph with protein/substrate nodes that have an edge if the pair is in the dataset
    """
    G = nx.Graph()

    proteins = df[protein_col].unique()
    substrates = df[substrate_col].unique()

    # nodes = proteins and substrates
    G.add_nodes_from(proteins, bipartite="protein")
    G.add_nodes_from(substrates, bipartite="substrate")

    # TODO: Add cluster informations
    # edges = if pairs occurs in dataset there is an edge
    for _, row in df.iterrows():
        p = row[protein_col]
        s = row[substrate_col]
        l = row[label_col]
        force = None
        if row["dataset"]=='ESP' or row["dataset"]=='EZS':
            force = True
        else:
            force = False
        G.add_edge(p, s, label=l, force_train=force)

    return G

def split_graph(G, train_frac=0.7, test_frac=0.15, reserve_frac = 0.1, random_state=42):
    """
        Perform split on graph level
    """
    val_frac = (1 - train_frac - test_frac)
    assert (1 - train_frac - test_frac - val_frac) <= 0.0001, "Fractions don't add to 1!"

    # --- GET ALL EDGES ---
    edges = list(G.edges(data=True))

    # TODO: split by clusters

    # --- 1) RESERVE A FIXED FRACTION OF EDGES BEFORE ANY SPLITTING ---
    reserved_edges, remaining_edges = reserve_edges_by_degree(G, reserve_frac=reserve_frac)

    # separate remaining edges into force_train vs non_force
    force_train_edges = [(u, v, l, t) for (u, v, l, t) in remaining_edges if t==True]
    non_force_edges = [(u, v, l, t) for (u, v, l, t) in remaining_edges if t==False]

    n_force_train = len(force_train_edges)
    n_reserved_test = len(reserved_edges)
    n_remaining = len(non_force_edges)
    if n_remaining == 0:
        n_remaining = 1

    total_edges = n_force_train + n_reserved_test + n_remaining
    desired_train_total_frac = train_frac  # e.g., 0.7
    desired_test_total_frac = test_frac  # e.g., 0.15

    # Already assigned edges
    current_train = n_force_train
    current_test = n_reserved_test

    # Remaining fraction for the remaining edges
    remaining_train_frac = max(0, (desired_train_total_frac * total_edges - current_train) / n_remaining)
    remaining_test_frac = max(0, (desired_test_total_frac * total_edges - current_test) / n_remaining)
    remaining_val_frac = 1.0 - remaining_train_frac - remaining_test_frac

    groups = defaultdict(list)
    for u, v, l, t in non_force_edges:
        groups[l].append((u, v, l, t))

    train_edges, val_edges, test_edges = [], [], []

    for label, group in groups.items():
        group = group[:]  # copy
        random.seed(random_state)
        random.shuffle(group)

        n = len(group)
        n_train = int(remaining_train_frac * n)
        n_val = int(remaining_val_frac * n)

        # select subset of edges = training pairs
        train_edges.extend(group[:n_train])
        # select subset of remaining edges = val pairs (unseen edges maybe seen nodes)
        val_edges.extend(group[n_train:n_train + n_val])
        # rest for evaluation
        test_edges.extend(group[n_train + n_val:])

    # --- 3) ADD RESERVED EDGES TO TEST SET AND FORCE TRAIN TO TRAIN SET---
    test_edges.extend(reserved_edges)
    train_edges.extend(force_train_edges)

    # split test and val data into
    #   unseen edges with seen nodes = C1
    #   unseen edges with one seen node = C2
    #   unseen edges with unseen nodes = C3
    (C1_edges_val, C2_edges_val, C3_edges_val,
     C1_edges_test, C2_edges_test, C3_edges_test) = create_c_sets(train_edges,val_edges,test_edges)

    return {"train": train_edges,
            "C1_val": C1_edges_val, "C2_val": C2_edges_val, "C3_val": C3_edges_val,
            "C1_test": C1_edges_test, "C2_test": C2_edges_test, "C3_test": C3_edges_test}

def reserve_edges_by_degree(G, reserve_frac=0.10):
    """
    Reserve edges based on degree heuristic.
    Returns reserved_edges (list of (u, v, label))
    and remaining_edges in the same format.
    """

    # 1) Node degrees
    neighbor_counts = dict(G.degree())

    # 2a) Build dataframe with edge features
    edge_data = []
    for u, v, data in G.edges(data=True): # data zB {'force_train': False, 'label': 1}
        deg_u = neighbor_counts.get(u, 0)
        deg_v = neighbor_counts.get(v, 0)
        sum_deg = deg_u + deg_v
        label = data.get("label")
        train = data.get("force_train")
        edge_data.append((u, v, deg_u, deg_v, sum_deg, label, train))

    df_edges = pd.DataFrame(edge_data,
        columns=["node_protein", "node_sub", "deg_protein",
                 "deg_sub", "sum_deg", "label", "force_train"]
    )
    # 2b) Separate force_train edges
    force_train_df = df_edges[df_edges["force_train"]].copy()
    non_force_df = df_edges[~df_edges["force_train"]].copy()

    # Sort edges ascending by sum of degrees
    df_edges_sorted = non_force_df.sort_values("sum_deg", ascending=True).reset_index(drop=True)

    # 3) Reserve edges based on 10% of lowest-degree edges
    n_total = len(df_edges_sorted)
    n_reserve = int(reserve_frac * n_total)

    reserved_df = df_edges_sorted.iloc[:n_reserve].copy()
    remaining_df_test = df_edges_sorted.iloc[n_reserve:].copy()
    remaining_df = pd.concat([remaining_df_test, force_train_df], ignore_index=True)

    # 4) Convert both to list-of-tuples format: (u, v, label, force_train)
    reserved_edges = list(zip(reserved_df["node_protein"],
                              reserved_df["node_sub"],
                              reserved_df["label"],
                              reserved_df["force_train"]))

    remaining_edges = list(zip(remaining_df["node_protein"],
                               remaining_df["node_sub"],
                               remaining_df["label"],
                               remaining_df["force_train"]))

    return reserved_edges, remaining_edges

def create_c_sets(train_edges, val_edges, test_edges):
    """
        Assign edges in evaluation set to:
            - C1: both gt and substrate seen in training
            - C2: one unseen (either gt or substrate)
            - C3: both unseen
    """
    train_seen = set()
    for u, v, l, t in train_edges:
        train_seen.add(u)
        train_seen.add(v)
    train_val_seen = set(train_seen)
    for u, v, l, t in val_edges:
        train_val_seen.add(u)
        train_val_seen.add(v)

    # C1/C2/C3 split
    C1_test, C2_test, C3_test = [], [], []
    C1_val, C2_val, C3_val = [], [], []

    for u, v, l, t in test_edges:
        u_seen = u in train_val_seen
        v_seen = v in train_val_seen

        if u_seen and v_seen: # both seen
            C1_test.append((u, v, l, t))
        elif u_seen or v_seen: # one unseen
            C2_test.append((u, v, l, t))
        else: # both unseen
            C3_test.append((u, v, l, t))

    for u, v, l, t in val_edges:
        u_seen = u in train_seen
        v_seen = v in train_seen

        if u_seen and v_seen: # both seen
            C1_val.append((u, v, l, t))
        elif u_seen or v_seen: # one unseen
            C2_val.append((u, v, l, t))
        else: # both unseen
            C3_val.append((u, v, l, t))

    return (
        C1_val, C2_val, C3_val,
        C1_test, C2_test, C3_test
    )


def check_split(train, c1_val, c2_val, c3_val, c1_test, c2_test, c3_test, protein_col, substrate_col):
    """
        Check if the data is split correctly and there is no data leakage
    """
    val_df = pd.concat([c1_val, c2_val, c3_val], axis=0)
    common_rows = pd.merge(train, val_df, how='inner')
    assert len(common_rows) == 0, "Train edge in validation set!"

    evaluation_df = pd.concat([c1_test, c2_test, c3_test], axis=0)
    common_rows_eval = pd.merge(train, evaluation_df, how='inner')
    assert len(common_rows_eval) == 0, "Train edge in evaluation set!"

    train_proteins = set(train[protein_col].unique())
    train_substrates = set(train[substrate_col].unique())
    val_proteins = set(val_df[protein_col].unique())
    val_substrates = set(val_df[substrate_col].unique())

    # ----------------------------
    #  VALIDATION C-SET CHECKS
    # ----------------------------

    # C1 consistency
    seen_proteins = train_proteins
    seen_substrates = train_substrates
    c1_proteins = set(c1_val[protein_col].unique())
    c1_substrates = set(c1_val[substrate_col].unique())
    assert c1_proteins <= seen_proteins, "C1 protein not in training!"
    assert c1_substrates <= seen_substrates, "C1 substrate not in training!"

    # C2 consistency
    for idx, row in c2_val.iterrows():
        p_seen = (row[protein_col] in train_proteins)
        s_seen = (row[substrate_col] in train_substrates)
        assert p_seen != s_seen, f"C2 split invalid for row {idx}"

    # C3 consistency
    c3_proteins = set(c3_val[protein_col].unique())
    c3_substrates = set(c3_val[substrate_col].unique())
    assert len(train_proteins & c3_proteins) == 0, "Data leakage: C3 protein in training!"
    assert len(train_substrates & c3_substrates) == 0, "Data leakage: C3 substrate in training!"

    # ----------------------------
    #  EVALUATION C-SET CHECKS
    # ----------------------------

    # C1 consistency
    seen_proteins = train_proteins | val_proteins
    seen_substrates = train_substrates | val_substrates
    c1_proteins = set(c1_test[protein_col].unique())
    c1_substrates = set(c1_test[substrate_col].unique())
    assert c1_proteins <= seen_proteins, "C1 protein not in training!"
    assert c1_substrates <= seen_substrates, "C1 substrate not in training!"

    # C2 consistency
    for idx, row in c2_test.iterrows():
        p_seen = (row[protein_col] in train_proteins) or (row[protein_col] in val_proteins)
        s_seen = (row[substrate_col] in train_substrates) or (row[substrate_col] in val_substrates)
        assert p_seen != s_seen, f"C2 split invalid for row {idx}"

    # C3 consistency
    c3_proteins = set(c3_test[protein_col].unique())
    c3_substrates = set(c3_test[substrate_col].unique())
    assert len(train_proteins & c3_proteins) == 0, "Data leakage: C3 protein in training!"
    assert len(train_substrates & c3_substrates) == 0, "Data leakage: C3 substrate in training!"
    assert len(val_proteins & c3_proteins) == 0, "Data leakage: C3 protein in validation set!"
    assert len(val_substrates & c3_substrates) == 0, "Data leakage: C3 substrate in validation set!"
