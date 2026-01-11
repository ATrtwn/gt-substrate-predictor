import pandas as pd
import random
import networkx as nx
from pathlib import Path

from src.utils.visualization import (plot_split_graph,
                                     plot_graph_connectivity,
                                     plot_connectivity_separate,
                                     plot_degree_distributions)

# data directory
data_dir = Path(__file__).parent.parent.parent / "data"

def get_clusters():
    cluster_tsv = data_dir / "GT_cluster_cluster.tsv"
    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Missing required .tsv file: {cluster_tsv}\n"
                                "Run clustering via:\n"
                                "tools\\mmseqs\\bin\\mmseqs.bat easy-cluster data\\UGT.fasta data\\GT_cluster tmp --min-seq-id 0.7 -c 0.7")

    df_clusters = pd.read_csv(
        cluster_tsv,
        sep="\t",
        header=None,
        names=["rep_id", "seq_id"],
        dtype=int
    )
    # Assign numeric cluster IDs
    rep_to_cluster = {
        rep: i for i, rep in enumerate(df_clusters["rep_id"].unique())
    }
    df_clusters["cluster_id"] = df_clusters["rep_id"].map(rep_to_cluster)

    return df_clusters


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
        u, v, l, c = edge_to_move
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
        u, v, l, c = edge_to_move
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
        for u, v, l, c in train:
            seen_nodes.add(u)
            seen_nodes.add(v)
        for u, v, l, c in val:
            seen_nodes.add(u)
            seen_nodes.add(v)
        # make plots
        plot_split_graph(graph, train, val, C1_test, C2_test, C3_test, seen_nodes)
        plot_graph_connectivity(graph)
        plot_connectivity_separate(graph)
        plot_degree_distributions(graph)

    # perform split on df based on graph split
    df_split = df.copy()
    df_split['split'] = ''
    df_split = df_split.sort_values(
        by=["UGT_ID", "substrate"],
        ascending=[True, True]
    ).reset_index(drop=True)

    def assign_split(row, edge_list, split_name):
        if row['split'] == '':
            if (row['UGT_ID'], row['substrate'], row['is_active']) in edge_list:
                return split_name
        return row['split']

    c1_val_pairs = [(u, v, l) for (u, v, l, c) in C1_val]
    c2_val_pairs = [(u, v, l) for (u, v, l, c) in C2_val]
    c3_val_pairs = [(u, v, l) for (u, v, l, c) in C3_val]
    c1_test_pairs = [(u, v, l) for (u, v, l, c) in C1_test]
    c2_test_pairs = [(u, v, l) for (u, v, l, c) in C2_test]
    c3_test_pairs = [(u, v, l) for (u, v, l, c) in C3_test]
    train_pairs = [(u, v, l) for (u, v, l, c) in train]

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

    # edges = if pairs occurs in dataset there is an edge
    for _, row in df.iterrows():
        p = row[protein_col]
        s = row[substrate_col]
        l = row[label_col]
        c = row['cluster_id']
        force = None
        if row["dataset"]=='ESP' or row["dataset"]=='EZS':
            force = True
        else:
            force = False
        G.add_edge(p, s, label=l, force_train=force, cluster=c)

    return G


def split_graph(G, train_frac=0.7, test_frac=0.15, reserve_frac=0.1, random_state=42):
    """
        Perform split on graph level
    """
    val_frac = (1 - train_frac - test_frac)
    assert (1 - train_frac - test_frac - val_frac) <= 0.0001, "Fractions don't add to 1!"

    # get all edges: edges = list(G.edges(data=True))
    # edges contain: ugt, substrate, label, force_train (ESP and EZS are not allowed in val/test sets), cluster id
    edges = [
        (u, v,
         data.get("label"),
         data.get("force_train"),
         data.get("cluster"))
        for u, v, data in G.edges(data=True)
    ]

    # separate remaining edges into force_train vs non_force
    force_train_edges = [(u, v, l, c) for (u, v, l, t, c) in edges if t == True]
    non_force_edges = [(u, v, l, c) for (u, v, l, t, c) in edges if t == False]

    # reserve a fixed fraction of edges before any splitting
    neighbor_counts = dict(G.degree())
    reserved_edges, remaining_edges = reserve_edges_by_degree(non_force_edges, neighbor_counts,
                                                              reserve_frac=reserve_frac)

    # determine split sizes
    n_force_train = len(force_train_edges)
    n_reserved_test = len(reserved_edges)
    n_remaining = len(remaining_edges)

    # Already assigned edges
    current_train = n_force_train
    current_test = n_reserved_test

    # assign clusters to train/val/test and balance labels (as good as possible)
    train_target = max(0, int(train_frac * len(edges) - current_train))
    test_target = max(0, int(test_frac * len(edges) - current_test))
    val_target = n_remaining - train_target - test_target

    train_edges, val_edges, test_edges = assign_clusters(remaining_edges,
                                                         train_target,
                                                         val_target,
                                                         test_target)

    # add reserved edges to test set and force train to train set
    test_edges.extend(reserved_edges)
    train_edges.extend(force_train_edges)

    # split test and val data into
    #   unseen edges with seen nodes = C1
    #   unseen edges with one seen node = C2
    #   unseen edges with unseen nodes = C3
    (C1_edges_val, C2_edges_val, C3_edges_val,
     C1_edges_test, C2_edges_test, C3_edges_test) = create_c_sets(train_edges, val_edges, test_edges)

    return {"train": train_edges,
            "C1_val": C1_edges_val, "C2_val": C2_edges_val, "C3_val": C3_edges_val,
            "C1_test": C1_edges_test, "C2_test": C2_edges_test, "C3_test": C3_edges_test}

def reserve_edges_by_degree(edges, neighbor_counts, reserve_frac=0.10):
    """
    Reserve edges based on degree heuristic.
    Returns reserved_edges (list of (u, v, label))
    and remaining_edges in the same format.
    """

    # Build dataframe with edge features
    edge_data = []
    for u, v, l, c in edges: # data zB {'force_train': False, 'label': 1}
        deg_u = neighbor_counts.get(u, 0)
        deg_v = neighbor_counts.get(v, 0)
        sum_deg = deg_u + deg_v
        edge_data.append((u, v, deg_u, deg_v, sum_deg, l, c))

    # Sort edges ascending by sum of degrees
    edge_data.sort(key=lambda e: e[4])  # sort by sum_deg (ascending)

    # Reserve edges based on 10% of lowest-degree edges
    n_total = len(edge_data)
    n_reserve = int(reserve_frac * n_total)

    reserved_raw = edge_data[:n_reserve]
    remaining_raw = edge_data[n_reserve:]

    # Convert both to list-of-tuples format: (u, v, data)
    reserved_edges = [
        (u, v, label, cluster)
        for (u, v, _, _, _, label, cluster) in reserved_raw
    ]

    remaining_edges = [
        (u, v, label, cluster)
        for (u, v, _, _, _, label, cluster) in remaining_raw
    ]

    return reserved_edges, remaining_edges

def assign_clusters(valid_edges, train_target, val_target, test_target):
    from collections import defaultdict

    cluster_stats = defaultdict(lambda: {
        "edges": [],
        "n": 0,
        "pos": 0,
        "neg": 0
    })

    for u, v, label, cluster in valid_edges:
        s = cluster_stats[cluster]
        s["edges"].append((u, v, label, cluster))
        s["n"] += 1
        if label == 1:
            s["pos"] += 1
        else:
            s["neg"] += 1

    targets = {
        "train": train_target,
        "val": val_target,
        "test": test_target
    }

    splits = {
        "train": {"edges": [], "n": 0, "pos": 0, "neg": 0},
        "val": {"edges": [], "n": 0, "pos": 0, "neg": 0},
        "test": {"edges": [], "n": 0, "pos": 0, "neg": 0},
    }

    clusters_sorted = sorted(
        cluster_stats.items(),
        key=lambda x: x[1]["n"],
        reverse=True
    )

    for cluster_id, stats in clusters_sorted:
        best_split = None
        best_score = float("inf")

        for name in ["test", "val", "train"]:
            score = score_assignment(splits[name], stats, targets[name])
            if score < best_score:
                best_score = score
                best_split = name

        # assign cluster
        splits[best_split]["edges"].extend(stats["edges"])
        splits[best_split]["n"] += stats["n"]
        splits[best_split]["pos"] += stats["pos"]
        splits[best_split]["neg"] += stats["neg"]

    train_edges = splits["train"]["edges"]
    val_edges = splits["val"]["edges"]
    test_edges = splits["test"]["edges"]

    return train_edges, val_edges, test_edges

def score_assignment(split, cluster, target):
    """
    Lower score = better assignment
    """
    # asymmetric size penalty
    after = split["n"] + cluster["n"]
    if after <= target:
        size_pen = (target - after) * 0.1
    else:
        size_pen = (after - target) * 2.0

    # label imbalance penalty
    total = split["n"] + cluster["n"]
    if total == 0:
        label_penalty = 0
    else:
        pos_frac = (split["pos"] + cluster["pos"]) / total
        label_penalty = abs(pos_frac - 0.5)

    penalty = size_pen + 0.5 * label_penalty * total

    return penalty

def create_c_sets(train_edges, val_edges, test_edges):
    """
        Assign edges in evaluation set to:
            - C1: both gt and substrate seen in training
            - C2: one unseen (either gt or substrate)
            - C3: both unseen
    """
    train_seen = set()
    for u, v, l, c in train_edges:
        train_seen.add(u)
        train_seen.add(v)
    train_val_seen = set(train_seen)
    for u, v, l, c in val_edges:
        train_val_seen.add(u)
        train_val_seen.add(v)

    # C1/C2/C3 split
    C1_test, C2_test, C3_test = [], [], []
    C1_val, C2_val, C3_val = [], [], []

    for u, v, l, c in test_edges:
        u_seen = u in train_val_seen
        v_seen = v in train_val_seen

        if u_seen and v_seen: # both seen
            C1_test.append((u, v, l, c))
        elif u_seen or v_seen: # one unseen
            C2_test.append((u, v, l, c))
        else: # both unseen
            C3_test.append((u, v, l, c))

    for u, v, l, c in val_edges:
        u_seen = u in train_seen
        v_seen = v in train_seen

        if u_seen and v_seen: # both seen
            C1_val.append((u, v, l, c))
        elif u_seen or v_seen: # one unseen
            C2_val.append((u, v, l, c))
        else: # both unseen
            C3_val.append((u, v, l, c))

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
