import json
import os
import random

from sklearn.model_selection import StratifiedKFold
import numpy as np
import networkx as nx

import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import contains_self_loops, contains_isolated_nodes, \
    is_undirected, to_networkx, degree

from collections import Counter


ROOT = '../../data'
DATASETS = ['MUTAG', 'NCI1', 'NCI109', 'PROTEINS', 'PTC_MR', 'ENZYMES']

def drop_edges(graph, drop_rate, seed=0):
    """
    Randomly removes a percentage of edges from the input graph.

    Args:
        graph (Data): Input PyG graph.
        drop_rate (float): Ratio of edges to drop.
        seed (int): Random seed for reproducibility.

    Returns:
        Data: Graph with dropped edges.
    """

    edge_index = graph.edge_index
    num_edges = edge_index.size(1)

    if drop_rate <= 0.0 or num_edges == 0:
        return graph  # No drop

    num_to_drop = int(num_edges * drop_rate)

    # Fix seed for reproducibility
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)

    keep_indices = perm[num_to_drop:]  # Indices to keep
    new_edge_index = edge_index[:, keep_indices]

    new_graph = graph.clone()
    new_graph.edge_index = new_edge_index

    return new_graph

def drop_graphs(graphs, drop_rate, seed=0):
    """
    Randomly selects and returns a subset of the input graph list.

    Args:
        graphs (list[Data]): List of graphs.
        drop_rate (float): Proportion of graphs to drop.
        seed (int): Random seed for reproducibility.

    Returns:
        list[Data]: Subset of input graphs.
    """

    if drop_rate <= 0.0 or len(graphs) == 0:
        return graphs  # No drop

    num_graphs = len(graphs)
    num_to_keep = int(num_graphs * (1.0 - drop_rate))

    rng = random.Random(seed)
    indices = list(range(num_graphs))
    rng.shuffle(indices)

    selected_indices = indices[:num_to_keep]
    return [graphs[i] for i in selected_indices]

def to_degree_features(data):
    """
    Generates one-hot node degree features for a list of graphs.

    Args:
        data (list[Data]): List of graphs.

    Returns:
        Tensor: One-hot encoded degree features for all nodes.
    """

    d_list = []
    for graph in data:
        d_list.append(degree(graph.edge_index[0], num_nodes=graph.num_nodes))
    x = torch.cat(d_list).long()
    unique_degrees = torch.unique(x)
    mapper = torch.full_like(x, fill_value=1000000000)
    mapper[unique_degrees] = torch.arange(len(unique_degrees))
    x_onehot = torch.zeros(x.size(0), len(unique_degrees))
    x_onehot[torch.arange(x.size(0)), mapper[x]] = 1
    return x_onehot

def compute_class_prior(data):
    """
    Computes the class prior (ratio of positive labels) for a dataset.

    Args:
        data (Dataset): PyG dataset.

    Returns:
        float: Proportion of positive class (label==1).
    """

    labels = [data[i].y.item() for i in range(len(data))]
    num_pos = sum(1 for y in labels if y == 1)
    return num_pos / len(labels)

def load_data(dataset, degree_x=True):
    """
    Loads a dataset from TUDataset and prepares node features.

    Args:
        dataset (str): Dataset name.
        degree_x (bool): If True, use one-hot node degrees as features.

    Returns:
        TUDataset: Loaded dataset with modified node features.
    """

    if dataset == 'Twitter':
        dataset = 'TWITTER-Real-Graph-Partial'
    data = TUDataset(root=os.path.join(ROOT, 'graphs'), name=dataset,
                     use_node_attr=False)
    data.data.edge_attr = None
    if data.num_node_features == 0:
        data.slices['x'] = torch.tensor([0] + data.data.num_nodes).cumsum(0)
        if degree_x:
            data.data.x = to_degree_features(data)
        else:
            num_all_nodes = sum(g.num_nodes for g in data)
            data.data.x = torch.ones((num_all_nodes, 1))

    if dataset == 'ENZYMES':
        labels = data.data.y.tolist()
        counter = Counter(labels)
        most_common_class, _ = counter.most_common(1)[0]

        binary_labels = [1 if y == most_common_class else 0 for y in labels]
        data.data.y = torch.tensor(binary_labels, dtype=torch.long)

    return data

def load_data_fold(dataset, fold, degree_x=True, observed_labeled_ratio=0.5, num_folds=10, seed=0, random_drop=0.0):
    """
    Loads a train/test split of the dataset and applies PU setting.

    Args:
        dataset (str): Dataset name.
        fold (int): Fold index (0~9).
        degree_x (bool): If True, use one-hot degree features.
        observed_labeled_ratio (float): Proportion of observed positives.
        num_folds (int): Number of folds for StratifiedKFold.
        seed (int): Random seed.
        random_drop (float): Edge drop rate.

    Returns:
        tuple: (List of PU-labeled training graphs, test graphs)
    """

    assert 0 <= fold < 10

    data = load_data(dataset, degree_x)
    path = os.path.join(ROOT, 'splits', dataset, f'{fold}.json')
    if not os.path.exists(path):
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        trn_idx, test_idx = list(skf.split(np.zeros(data.len()), data.data.y))[fold]
        trn_idx = [int(e) for e in trn_idx]
        test_idx = [int(e) for e in test_idx]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(dict(training=trn_idx, test=test_idx), f, indent=4)

    with open(path) as f:
        indices = json.load(f)
    trn_graphs = [data[i] for i in indices['training']]
    test_graphs = [data[i] for i in indices['test']]

    print(f"# pos in original train: {sum(g.y.item() == 1 for g in trn_graphs)} / {len(trn_graphs)}")
    pu_trn_graphs = convert_to_pu_setting(trn_graphs, observed_labeled_ratio)

    if random_drop > 0.0:
        pu_trn_graphs = [drop_edges(g, random_drop, seed=seed + i) for i, g in enumerate(pu_trn_graphs)]
        test_graphs = [drop_edges(g, random_drop, seed=seed + 1000 + i) for i, g in enumerate(test_graphs)]
        # pu_trn_graphs = drop_graphs(pu_trn_graphs, drop_rate=random_drop, seed=seed)
        # test_graphs = drop_graphs(test_graphs, drop_rate=random_drop, seed=seed + 1000)

    print(f"# pos in modified train: {sum(g.y.item() == 1 for g in pu_trn_graphs)} / {len(pu_trn_graphs)}")
    print(f"# pos in test: {sum(g.y.item() == 1 for g in test_graphs)} / {len(test_graphs)}")

    return pu_trn_graphs, test_graphs


def convert_to_pu_setting(graphs, observed_label_ratio=0.5, seed=0):
    """
    Converts a labeled dataset into a PU setting by hiding a portion of positive labels.

    Args:
        graphs (list[Data]): List of labeled graphs.
        observed_label_ratio (float): Ratio of positives to keep labeled.
        seed (int): Random seed.

    Returns:
        list[Data]: Modified list with partial positive labels.
    """

    rng = random.Random(seed)

    # Identify positive indices
    pos_indices = [i for i, g in enumerate(graphs) if g.y.item() == 1]
    rng.shuffle(pos_indices)

    num_observed = int(len(pos_indices) * observed_label_ratio)
    observed_pos_indices = set(pos_indices[:num_observed])

    for i, g in enumerate(graphs):
        if g.y.item() == 1 and i not in observed_pos_indices:
            g.y = torch.tensor([0])
        else:
            g.y = torch.tensor([1]) if g.y.item() == 1 else torch.tensor([0])

    return graphs


def is_connected(graph):
    """
    Checks whether the input graph is connected.

    Args:
        graph (Data): Input PyG graph.

    Returns:
        bool: True if the graph is connected.
    """

    return nx.is_connected(to_networkx(graph, to_undirected=True))


def print_stats():
    """
    Prints statistics for each dataset, including graph counts and structural properties.
    """

    for data in DATASETS:
        out = load_data(data)
        num_graphs = len(out)
        num_nodes = out.data.x.size(0)
        num_edges = out.data.edge_index.size(1) // 2
        num_features = out.num_features
        num_classes = out.num_classes
        print(f'{data}\t{num_graphs}\t{num_nodes}\t{num_edges}\t{num_features}\t'
              f'{num_classes}', end='\t')

        undirected, self_loops, onehot, connected, isolated_nodes = \
            True, False, True, True, False
        for graph in out:
            if not is_undirected(graph.edge_index, num_nodes=graph.num_nodes):
                undirected = False
            if contains_self_loops(graph.edge_index):
                self_loops = True
            if ((graph.x > 0).sum(dim=1) != 1).sum() > 0:
                onehot = False
            if not is_connected(graph):
                connected = False
            if contains_isolated_nodes(graph.edge_index, num_nodes=graph.num_nodes):
                isolated_nodes = True
        print(f'{undirected}\t{self_loops}\t{onehot}\t{connected}\t{isolated_nodes}')


def download():
    """
    Downloads all datasets and precomputes the 10-fold splits for each.
    """

    for data in DATASETS:
        load_data(data)
        for fold in range(10):
            load_data_fold(data, fold)


if __name__ == '__main__':
    download()
    print_stats()
