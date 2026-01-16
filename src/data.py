import json
import os
import random

from sklearn.model_selection import StratifiedKFold
import numpy as np
import networkx as nx

import torch
import torch.nn.functional as F
from torch_geometric.datasets import (
    TUDataset,
    UPFD,
    BA2MotifDataset,
    BAMultiShapesDataset,
)
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import contains_self_loops, contains_isolated_nodes, \
    is_undirected, to_networkx, degree

from collections import Counter

ROOT = '../data'
DATASETS = ['DD', 'MUTAG', 'NCI1', 'NCI109', 'PROTEINS', 'PTC_MR',
            'COLLAB', 'Twitter', 'ENZYMES',
            'BA2Motif', 'BAMultiShapes',]

def drop_edges(graph, drop_rate, seed=0):
    """Return a copy of the graph with a percentage of edges randomly removed."""
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
    """Return a subset of graphs after randomly dropping a portion of them."""
    if drop_rate <= 0.0 or len(graphs) == 0:
        return graphs  # No drop

    num_graphs = len(graphs)
    num_to_keep = int(num_graphs * (1.0 - drop_rate))

    rng = random.Random(seed)
    indices = list(range(num_graphs))
    rng.shuffle(indices)

    selected_indices = indices[:num_to_keep]
    return [graphs[i] for i in selected_indices]

def to_degree_features(data, cap=64):
    xs = []
    for g in data:
        d = degree(g.edge_index[0], num_nodes=g.num_nodes).long()
        d = torch.clamp(d, max=cap)                 # 과도한 차수 캡
        x = F.one_hot(d, num_classes=cap+1).float() # (N, cap+1)
        xs.append(x)
    X = torch.cat(xs, dim=0).contiguous()
    return X

def compute_class_prior(data):
    labels = [data[i].y.item() for i in range(len(data))]
    num_pos = sum(1 for y in labels if y == 1)
    return num_pos / len(labels)

def _parse_dataset_spec(spec: str):
    """spec을 해석해 family와 인자를 반환."""
    low = spec.lower()
    if low in ('ba2motif', 'ba-2motif', 'ba2'):
        return ('BA2Motif', {})
    if low in ('bamultishapes', 'ba-multishapes', 'bamulti'):
        return ('BAMultiShapes', {})
    return ('TU', {'name': spec})

class _ListInMemoryDataset(InMemoryDataset):
    """여러 split을 합친 Data list를 InMemoryDataset로 묶어줌."""
    def __init__(self, data_list):
        super().__init__(root='')
        self.data, self.slices = self.collate(data_list)

def load_data(dataset, degree_x=True):
    family, args = _parse_dataset_spec(dataset)

    if family == 'TU':
        name = args['name']
        if name == 'Twitter':
            name = 'TWITTER-Real-Graph-Partial'
        data = TUDataset(root=os.path.join(ROOT, 'graphs'), name=name, use_node_attr=False)
        data.data.edge_attr = None

        # 노드 특성이 없으면 degree one-hot 또는 상수 벡터 생성
        if data.num_node_features == 0:
            # 그래프별 노드 수 리스트로 x 슬라이스 생성
            node_counts = [g.num_nodes for g in data]
            data.slices['x'] = torch.tensor([0] + node_counts).cumsum(0)

            if degree_x:
                data.data.x = to_degree_features(data)  # 아래 개선 버전 쓰길 권장
            else:
                num_all_nodes = sum(node_counts)
                data.data.x = torch.ones((num_all_nodes, 1), dtype=torch.float32)

        # ENZYMES는 다중클래스라 기존 코드처럼 이진으로 변환
        if name == 'ENZYMES':
            labels = data.data.y.tolist()
            from collections import Counter
            counter = Counter(labels)
            most_common_class, _ = counter.most_common(1)[0]
            binary_labels = [1 if y == most_common_class else 0 for y in labels]
            data.data.y = torch.tensor(binary_labels, dtype=torch.long)

        return data

    elif family == 'UPFD':
        # 모든 split(train/val/test)을 합쳐서 10-fold로 재분할
        upfd_root = os.path.join(ROOT, 'UPFD')
        all_list = []
        for split in ('train', 'val', 'test'):
            ds = UPFD(root=upfd_root, name=args['name'], feature=args['feature'], split=split)
            # ds는 InMemoryDataset이므로 아이템을 뽑아 리스트로 결합
            all_list.extend([ds[i] for i in range(len(ds))])
        data = _ListInMemoryDataset(all_list)
        data.data.edge_attr = None  # 일관성 유지
        return data

    elif family == 'BA2Motif':
        # 1000개 그래프, 클래스 2 (House vs Cycle) — 바로 사용 가능
        # 문서 상 통계: features=10, classes=2 :contentReference[oaicite:6]{index=6}
        return BA2MotifDataset(root=os.path.join(ROOT, 'BA2Motif'))

    elif family == 'BAMultiShapes':
        # 1000개 그래프, 클래스 2 (모티프 조합 논리식 기반) — 바로 사용 가능
        # 문서 상 통계: features=10, classes=2 :contentReference[oaicite:7]{index=7}
        return BAMultiShapesDataset(root=os.path.join(ROOT, 'BAMultiShapes'))

    else:
        raise ValueError(f"Unknown dataset spec: {dataset}")

def load_data_fold(dataset, fold, degree_x=True, observed_labeled_ratio=0.5, num_folds=10, seed=0, random_drop=0.0):
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
    return nx.is_connected(to_networkx(graph, to_undirected=True))


def print_stats():
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
    for data in DATASETS:
        load_data(data)
        for fold in range(10):
            load_data_fold(data, fold)


if __name__ == '__main__':
    download()
    print_stats()
