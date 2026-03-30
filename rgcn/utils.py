import numpy as np
import random
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs
import torch
from math import sqrt
from typing import Union


# convert the inputs from cpu to gpu, accelerate the running speed
def convert_to_gpu(*data, device: str):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def set_random_seed(seed: int = 0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_model(model: nn.Module, model_path: str):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def get_n_params(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: model
    :return: int
    """
    return sum(p.numel() for p in model.parameters())


def load_patent_dataset(data_path: str, predict_category: str, data_split_idx_path: str = None):
    """
    load dataset
    :param data_path: data file path
    :param predict_category: predict node category
    :param data_split_idx_path: split index file path
    :return:
    """
    graph_list, labels = load_graphs(data_path)

    split_idx = torch.load(data_split_idx_path)
    train_idx, valid_idx, test_idx = split_idx['train'][predict_category], split_idx['test'][predict_category]
    if not isinstance(train_idx, np.ndarray):
        train_idx = np.array(train_idx)
    if not isinstance(test_idx, np.ndarray):
        test_idx = np.array(test_idx)
    print(f"train nodes:{len(train_idx)}, test nodes:{len(test_idx)}")
    return graph_list[0], train_idx, test_idx


def load_dataset(data_path: str, predict_category: str, data_split_idx_path: str = None):
    """
    load dataset
    :param data_path: data file path
    :param predict_category: predict node category
    :param data_split_idx_path: split index file path
    :return:
    """
    graph_list, labels = load_graphs(data_path)

    graph = graph_list[0]

    if isinstance(labels, dict):
        labels = labels[predict_category].squeeze(dim=-1)
    print("number of nodes:", len(labels))

    num_classes = len(labels.unique())

    split_idx = torch.load(data_split_idx_path, map_location='cpu')
    train_idx, test_idx = split_idx['train'][predict_category], split_idx['test'][predict_category]
    print(f"train nodes:{len(train_idx)}, test nodes:{len(test_idx)}")
    return graph, labels, num_classes, train_idx, test_idx


def get_node_data_loader(node_neighbors_min_num: int, n_layers: int,
                         graph: dgl.DGLGraph, batch_size: int, sampled_node_type: str,
                         train_idx: torch.Tensor, test_idx: torch.Tensor,
                         shuffle: bool = True, drop_last: bool = True, num_workers: int = 4):
    """
    get graph node data loader, including train_loader, val_loader and test_loader
    :return:
    """
    # list of neighbors to sample per edge type for each GNN layer
    sample_nodes_num = []
    for layer in range(n_layers):
        sample_nodes_num.append({etype: node_neighbors_min_num + layer for etype in graph.canonical_etypes})

    # neighbor sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)

    train_loader = dgl.dataloading.DataLoader(
        graph, {sampled_node_type: train_idx}, sampler,
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    test_loader = dgl.dataloading.DataLoader(
        graph, {sampled_node_type: test_idx}, sampler,
        batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    return train_loader, test_loader


def get_predict_edge_index(graph: dgl.DGLGraph, sampled_edge_type: Union[str, tuple],
                           sample_edge_rate: float, seed: int = 0):
    """
    get predict edge index, return train_edge_idx, valid_edge_idx, test_edge_idx
    :return:
    """
    torch.manual_seed(seed=seed)

    selected_edges_num = int(graph.number_of_edges(sampled_edge_type) * sample_edge_rate)
    permute_idx = torch.randperm(graph.number_of_edges(sampled_edge_type))

    train_edge_idx = permute_idx[: 3 * selected_edges_num]
    test_edge_idx = permute_idx[4 * selected_edges_num: 5 * selected_edges_num]

    return train_edge_idx, test_edge_idx


def get_edge_data_loader(node_neighbors_min_num: int, n_layers: int,
                         graph: dgl.DGLGraph, batch_size: int, sampled_edge_type: str,
                         negative_sample_edge_num: int,
                         train_edge_idx: torch.Tensor, 
                         test_edge_idx: torch.Tensor,
                         reverse_etypes: dict, shuffle: bool = True, drop_last: bool = False,
                         num_workers: int = 4):
    """
    get edge data loader for link prediction, including train_loader, val_loader and test_loader
    :return:
    """
    # list of neighbors to sample per edge type for each GNN layer
    sample_nodes_num = []
    for layer in range(n_layers):
        sample_nodes_num.append({etype: node_neighbors_min_num + layer for etype in graph.canonical_etypes})

    # neighbor sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)
    train_neg_sampler = dgl.dataloading.negative_sampler.Uniform(negative_sample_edge_num)

    train_loader = dgl.dataloading.EdgeDataLoader(
        graph, {sampled_edge_type: train_edge_idx}, sampler, negative_sampler=train_neg_sampler, exclude='reverse_types',
        reverse_etypes=reverse_etypes,
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    # sample the same number of edges when evaluating the model, set negative number to 1
    eval_neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    test_loader = dgl.dataloading.EdgeDataLoader(
        graph, {sampled_edge_type: test_edge_idx}, sampler, negative_sampler=eval_neg_sampler, exclude='reverse_types',
        reverse_etypes=reverse_etypes,
        batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return train_loader, test_loader


def get_optimizer_and_lr_scheduler(model: nn.Module, optimizer_name: str, learning_rate: float, weight_deacy: float, steps_per_epoch: int, epochs: int):
    """
    get optimizer and lr scheduler
    :param model:
    :param optimizer_name:
    :param learning_rate:
    :param weight_deacy:
    :param steps_per_epoch:
    :param epochs:
    :return:
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_deacy, momentum=0.01)
    else:
        raise ValueError(f"wrong value for optimizer {optimizer_name}!")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps_per_epoch * epochs, eta_min=learning_rate / 100)

    return optimizer, scheduler

