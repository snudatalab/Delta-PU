from sklearn.metrics import f1_score, accuracy_score
from copy import deepcopy
import numpy as np

import torch
from torch_geometric.loader import DataLoader

from .loss import *
from .utils import *


def train_model(args, model, loss_func, optimizer, scheduler, trn_loader, test_loader, trn_graphs, prior, device):
    """
    Train a GNN model for one complete training cycle.

    Args:
        args (Namespace): Command-line arguments.
        model (nn.Module): GNN model to train.
        loss_func (Callable): Loss function (e.g., CrossEntropy or PU-aware loss).
        optimizer (Optimizer): Optimizer for parameter update.
        scheduler (Scheduler): Learning rate scheduler.
        trn_loader (DataLoader): DataLoader for training set.
        test_loader (DataLoader): DataLoader for test set.
        trn_graphs (List[Data]): List of training graph data.
        prior (float): Class prior for positive label.
        device (torch.device): Device for computation.

    Returns:
        Dict[str, List[float]]: Training/test loss, accuracy, and F1-score logs per epoch.
    """

    if args.verbose > 0:
        print(' epochs\t   loss trn_acc test_acc trn_f1 test_f1')

    out_list = dict(trn_loss=[], trn_acc=[], test_loss=[], test_acc=[], trn_f1=[], test_f1=[])

    if args.loss == 'dump':
        hop_weights = (torch.ones((len(trn_graphs), 5))/5).to(device)

    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0

        if args.loss == 'dump':
            instance_weights, val_mask = compute_instance_weights(model, loss_func, trn_graphs, device, prior, lr=args.meta_lr,
                                                        hop_weights=hop_weights)
            hop_weights = compute_hop_weights(model, loss_func, trn_graphs, device, val_mask, lr=args.meta_lr,
                                              instance_weights=instance_weights)
            model.inst_w = instance_weights
        else:
            instance_weights = None
            hop_weights = None

        for data in trn_loader:
            data = data.to(device)

            if args.loss == 'dump':
                inst_w = instance_weights[data.idx].to(device) if hasattr(data, 'idx') else None
                hop_w = hop_weights[data.idx].to(device) if hasattr(data, 'idx') else None

                output = model(data.x, data.edge_index, data.batch, hop_weights=hop_w)
                loss = loss_func(output, data.y, instance_weight=inst_w)
            else:
                output = model(data.x, data.edge_index, data.batch)
                loss = loss_func(output, data.y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()

        if args.schedule:
            scheduler.step()

        trn_loss = loss_sum / len(trn_loader)
        trn_acc, trn_f1 = eval_model(model, trn_loader, device)
        test_loss = eval_loss(model, loss_func, test_loader, device)
        test_acc, test_f1 = eval_model(model, test_loader, device)

        out_list['trn_loss'].append(trn_loss)
        out_list['trn_acc'].append(trn_acc)
        out_list['trn_f1'].append(trn_f1)
        out_list['test_loss'].append(test_loss)
        out_list['test_acc'].append(test_acc)
        out_list['test_f1'].append(test_f1)

        if args.verbose > 0 and (epoch + 1) % args.verbose == 0:
            print(f'{epoch + 1:7d}\t{trn_loss:7.4f}\t{trn_acc:7.4f}\t{test_acc:7.4f}\t{trn_f1:7.4f}\t{test_f1:7.4f}')

    return out_list

def train_iterations(args, model, loss_func, test_loader, trn_graphs, prior, device, num_iterations=5):
    """
    Iteratively refine model and labels using pseudo-labeling and meta-reweighting.

    Args:
        args (Namespace): Command-line arguments.
        model (nn.Module): GNN model.
        loss_func (Callable): Initial loss function for iteration 0.
        test_loader (DataLoader): DataLoader for test set.
        trn_graphs (List[Data]): Training graph data list.
        prior (float): Class prior for positive label.
        device (torch.device): Device for computation.
        num_iterations (int): Number of refinement iterations.

    Returns:
        List[Dict[str, List[float]]]: List of performance logs from each iteration.
    """

    all_results = []

    current_graphs = deepcopy(trn_graphs)
    for iteration in range(num_iterations):
        print(f"\n[ Iteration {iteration + 1} / {num_iterations} ]")

        # 1. prepare data loader
        trn_loader = DataLoader(current_graphs, batch_size=args.batch_size)

        # 2. initialize optimizer/loss/model (or reuse)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

        # 3. train model for 100 epochs
        if iteration == 0:
            args.loss = 'ce'
            loss_func = balancedCELoss()
        else:
            args.loss = 'dump'
            loss_func = WeightedLoss(class_prior=prior)

        results = train_model(args, model, loss_func, optimizer, scheduler, trn_loader, test_loader, current_graphs,
                              prior, device)
        all_results.append(results)

        if iteration == 0:
            continue

        change_cnt = 0
        current_graphs = deepcopy(trn_graphs)
        model.eval()
        for i, data in enumerate(current_graphs):
            data = data.to(device)

            with torch.no_grad():
                output = model(data.x, data.edge_index, data.batch)
            prob = torch.softmax(output, dim=-1)[0, 1].item()  # P(y=1)

            if data.y.item() == 0:  # unlabeled (originally)
                if prob > 0.95 and model.inst_w[i] == torch.min(model.inst_w):
                    data.y = torch.tensor([1]).long().to(device)
                    change_cnt += 1
                data.observed = 0
            else:
                data.observed = 1

    return all_results

@torch.no_grad()
def eval_model(model, loader, device):
    """
    Evaluate the model using accuracy and macro-F1 score.

    Args:
        model (nn.Module): Trained GNN model.
        loader (DataLoader): DataLoader for evaluation set.
        device (torch.device): Device for computation.

    Returns:
        Tuple[float, float]: Accuracy and macro-F1 score.
    """

    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        y_pred.append(output.argmax(dim=1).cpu())
        y_true.append(data.y.cpu())
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

@torch.no_grad()
def eval_loss(model, loss_func, loader, device):
    """
    Compute average loss over the given dataset.

    Args:
        model (nn.Module): Trained GNN model.
        loss_func (Callable): Loss function.
        loader (DataLoader): DataLoader for evaluation set.
        device (torch.device): Device for computation.

    Returns:
        float: Average loss value.
    """

    model.eval()
    count_sum, loss_sum = 0, 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        loss = loss_func(output, data.y).item()
        loss_sum += loss * len(data.y)
        count_sum += len(data.y)
    return loss_sum / count_sum
