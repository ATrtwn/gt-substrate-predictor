import pandas as pd
import random
import networkx as nx
from collections import defaultdict, Counter
from sklearn.model_selection import train_test_split
from src.utils.visualization import plot_split_graph, plot_graph_connectivity

def stratified_split_by_entities(df, protein_col="UGT_ID", substrate_col="substrate", random_state=42):
    """
    Perform stratified split according to generalization classes (C1, C2, C3).

    Splits data such that:
        - C1: both gt and substrate seen in training
        - C2: one unseen (either gt or substrate)
        - C3: both unseen
    """
    #TODO: consider distribution of GT families and chemical diversity!

    graph = create_graph(df,
                         protein_col=protein_col,
                         substrate_col=substrate_col)


    # split and re-balance if evaluation sets are too small
    edge_split = split_graph(graph, train_frac=0.7, test_frac=0.15, random_state=random_state)
    # balance sets, by moving single components to unseen
    train_edges = edge_split['train']
    val_edges = edge_split['val']
    C1_edges = edge_split["C1"]
    C2_edges = edge_split["C2"]
    C3_edges = edge_split['C3']

    while((len(C2_edges) < 50)):
        # choose an edge from c1
        random.seed(random_state)
        edge_to_move = random.choice(C1_edges)
        u, v, l = edge_to_move
        # choose which component to move from seen to unseen
        unseen_u = bool(random.getrandbits(1))
        if unseen_u:
            unseen_component = u
        else:
            unseen_component = v
        # remove now unseen components from train and val sets
        train_removed = [e for e in train_edges if unseen_component in e]
        train_edges = [e for e in train_edges if unseen_component not in e]
        val_removed = [e for e in val_edges if unseen_component in e]
        val_edges = [e for e in val_edges if unseen_component not in e]
        # re-assign pairs in evaluation sets
        test_edges = C1_edges + C2_edges + C3_edges + train_removed + val_removed
        C1_edges, C2_edges, C3_edges = create_evaluation_sets(train_edges, val_edges, test_edges)

    ### plot the split graph
    seen_nodes = set()
    for u, v, l in train_edges:
        seen_nodes.add(u)
        seen_nodes.add(v)
    plot_split_graph(graph, train_edges, val_edges, C1_edges, C2_edges, C3_edges, seen_nodes)
    plot_graph_connectivity(graph)
    ###

    # perform split on df based on graph split
    c1_set =  {(u, v) for (u, v, l) in C1_edges}
    c1 = df[df.apply(lambda row: (row[protein_col], row[substrate_col]) in c1_set, axis=1)].copy()
    c2_set = {(u, v) for (u, v, l) in C2_edges}
    c2 = df[df.apply(lambda row: (row[protein_col], row[substrate_col]) in c2_set, axis=1)].copy()
    c3_set = {(u, v) for (u, v, l) in C3_edges}
    c3 = df[df.apply(lambda row: (row[protein_col], row[substrate_col]) in c3_set, axis=1)].copy()
    train_set = {(u, v) for (u, v, l) in train_edges}
    train_df = df[df.apply(lambda row: (row[protein_col], row[substrate_col]) in train_set, axis=1)].copy()
    val_set = {(u, v) for (u, v, l) in val_edges}
    val_df = df[df.apply(lambda row: (row[protein_col], row[substrate_col]) in val_set, axis=1)].copy()

    return {"train": train_df, "val": val_df , "C1": c1, "C2": c2, "C3": c3}

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
        G.add_edge(p, s, label=l)

    return G

def reserve_edges_by_degree(G, reserve_frac=0.10):
    """
    Reserve edges based on degree heuristic.
    Returns reserved_edges (list of (u, v, label)) 
    and remaining_edges in the same format.
    """

    # 1) Node degrees
    neighbor_counts = dict(G.degree())

    # 2) Build dataframe with edge features
    edge_data = []
    for u, v, data in G.edges(data=True):
        deg_u = neighbor_counts.get(u, 0)
        deg_v = neighbor_counts.get(v, 0)
        sum_deg = deg_u + deg_v
        label = data.get("label")
        edge_data.append((u, v, deg_u, deg_v, sum_deg, label))

    df_edges = pd.DataFrame(edge_data, 
        columns=["node_protein", "node_sub", "deg_protein", 
                 "deg_sub", "sum_deg", "label"]
    )

    # Sort edges ascending by sum of degrees
    df_edges_sorted = df_edges.sort_values("sum_deg", ascending=True).reset_index(drop=True)

    # 3) Reserve edges based on 10% of lowest-degree edges
    n_total = len(df_edges_sorted)
    n_reserve = int(reserve_frac * n_total)

    reserved_df = df_edges_sorted.iloc[:n_reserve].copy()
    remaining_df = df_edges_sorted.iloc[n_reserve:].copy()

    # 4) Convert both to list-of-tuples format: (u, v, label)
    reserved_edges = list(zip(reserved_df["node_protein"],
                              reserved_df["node_sub"],
                              reserved_df["label"]))

    remaining_edges = list(zip(remaining_df["node_protein"],
                               remaining_df["node_sub"],
                               remaining_df["label"]))

    return reserved_edges, remaining_edges


def split_graph(G, train_frac=0.7, test_frac=0.15, reserve_frac = 0.1, random_state=42):
    """
        Perform split on graph level
    """
    val_frac = (1 - train_frac - test_frac)
    assert (1 - train_frac - test_frac - val_frac) <= 0.0001, "Fractions don't add to 1!"

    # --- GET ALL EDGES ---
    edges = list(G.edges(data=True))

    # --- 1) RESERVE A FIXED FRACTION OF EDGES BEFORE ANY SPLITTING ---
    reserved_edges, remaining_edges = reserve_edges_by_degree(G, reserve_frac=reserve_frac)

    groups = defaultdict(list)
    for u, v, l in remaining_edges:
        label = l
        groups[label].append((u, v, label))

    train_edges, val_edges, test_edges = [], [], []

    for label, group in groups.items():
        group = group[:]  # copy
        random.seed(random_state)
        random.shuffle(group)

        n = len(group)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)

        # random select subset of edges = training pairs
        train_edges.extend(group[:n_train])
        # random select subset of remaining edges = val pairs (unseen edges maybe seen nodes)
        val_edges.extend(group[n_train:n_train + n_val])
        # rest for evaluation
        test_edges.extend(group[n_train + n_val:])

    # --- 3) ADD RESERVED EDGES TO TEST SET ---
    test_edges.extend(reserved_edges)

    # split the rest into
    #   unseen edges with seen nodes = C1
    #   unseen edges with one seen node = C2
    #   unseen edges with unseen nodes = C3
    C1_edges, C2_edges, C3_edges = create_evaluation_sets(train_edges, val_edges, test_edges)

    return {"train": train_edges, "val": val_edges, "C1": C1_edges, "C2": C2_edges, "C3": C3_edges}

def create_evaluation_sets(train_edges, val_edges, test_edges):
    """
        Assign edges in evaluation set to:
            - C1: both gt and substrate seen in training
            - C2: one unseen (either gt or substrate)
            - C3: both unseen
    """
    seen_nodes = set()
    for u, v, l in train_edges:
        seen_nodes.add(u)
        seen_nodes.add(v)

    # C1/C2/C3 split
    C1_edges = []
    C2_edges = []
    C3_edges = []

    for u, v, l in test_edges:
        u_seen = u in seen_nodes
        v_seen = v in seen_nodes

        if u_seen and v_seen: # both seen
            C1_edges.append((u, v, l))
        elif u_seen or v_seen: # one unseen
            C2_edges.append((u, v, l))
        else: # both unseen
            C3_edges.append((u, v, l))

    return C1_edges,C2_edges,C3_edges

def check_split(train_df, val_df, c1_df, c2_df, c3_df, protein_col, substrate_col):
    """
        Check if the data is split correctly and there is no data leakage
    """
    common_rows = pd.merge(train_df, val_df, how='inner')
    assert len(common_rows) == 0, "Train edge in validation set!"

    evaluation_df = pd.concat([c1_df, c2_df, c3_df], axis=0)
    common_rows_eval = pd.merge(train_df, evaluation_df, how='inner')
    assert len(common_rows_eval) == 0, "Train edge in evaluation set!"

    train_proteins = set(train_df[protein_col].unique())
    train_substrates = set(train_df[substrate_col].unique())

    # C1 consistency
    c1_proteins = set(c1_df[protein_col].unique())
    c1_substrates = set(c1_df[substrate_col].unique())
    assert c1_proteins <= train_proteins, "C1 protein not in training!"
    assert c1_substrates <= train_substrates, "C1 substrate not in training!"

    # C2 consistency
    for idx, row in c2_df.iterrows():
        p_seen = row[protein_col] in train_proteins
        s_seen = row[substrate_col] in train_substrates
        assert p_seen != s_seen, f"C2 split invalid for row {idx}"

    # C3 consistency
    c3_proteins = set(c3_df[protein_col].unique())
    c3_substrates = set(c3_df[substrate_col].unique())
    assert len(train_proteins & c3_proteins) == 0, "Data leakage: C3 protein in training!"
    assert len(train_substrates & c3_substrates) == 0, "Data leakage: C3 substrate in training!"
