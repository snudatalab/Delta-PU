import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Batch
from torch.autograd import grad
from copy import deepcopy


class balancedCELoss(nn.Module):
    """
    Balanced cross-entropy loss that separates positive and unlabeled samples.

    Returns:
        Tensor: Either full instance-wise losses or mean loss for positive/unlabeled samples.
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, reduction='mean'):
        pos_mask = (target == 1)
        unl_mask = (target == 0)

        input_pos = pred[pos_mask]
        input_unl = pred[unl_mask]

        loss_pos = self.ce(input_pos, torch.ones_like(target[pos_mask])) if len(input_pos) > 0 else torch.tensor(0.0)
        loss_unl = self.ce(input_unl, torch.zeros_like(target[unl_mask])) if len(input_unl) > 0 else torch.tensor(0.0)

        full_loss = torch.zeros_like(target, dtype=torch.float, device=target.device)
        full_loss[pos_mask] = loss_pos
        full_loss[unl_mask] = loss_unl

        if reduction == 'none':
            return full_loss
        else:
            return loss_pos.mean() + loss_unl.mean()


class WeightedLoss(nn.Module):
    """
    Weighted loss function for PU learning that supports instance-wise reweighting.

    Args:
        class_prior (float): Estimated class prior (P(y=1)).
    """

    def __init__(self, class_prior):
        super().__init__()
        self.class_prior = class_prior
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred, target, instance_weight=None, reduction='mean'):
        """
        Compute loss with optional instance weights for positive and unlabeled samples.

        Args:
            pred (Tensor): Logits for each sample.
            target (Tensor): Ground-truth binary labels (0: unlabeled, 1: positive).
            instance_weight (Tensor, optional): Per-instance weights.
            reduction (str): 'mean' or 'none'.

        Returns:
            Tensor: Scalar loss or loss per instance.
        """

        pos_mask = (target == 1)
        unl_mask = (target != 1)

        input_pos = pred[pos_mask]
        input_unl = pred[unl_mask]

        loss_pos = self.ce(input_pos, torch.ones_like(target[pos_mask])) if len(input_pos) > 0 else torch.tensor(0.0)
        loss_unl = self.ce(input_unl, torch.zeros_like(target[unl_mask])) if len(input_unl) > 0 else torch.tensor(0.0)

        full_loss = torch.zeros_like(target, dtype=torch.float, device=target.device)
        full_loss[pos_mask] = loss_pos
        full_loss[unl_mask] = loss_unl

        if reduction == 'none':
            return full_loss
        elif reduction == 'mean' and instance_weight is not None:
            weight_pos = instance_weight[pos_mask] if len(input_pos) > 0 else torch.tensor(0.0)
            weight_unl = instance_weight[unl_mask] if len(input_unl) > 0 else torch.tensor(0.0)
        else:
            weight_pos = torch.ones_like(loss_pos) if len(input_pos) > 0 else torch.tensor(0.0)
            weight_unl = torch.ones_like(loss_unl) if len(input_unl) > 0 else torch.tensor(0.0)

        return (loss_pos * weight_pos).mean() + (loss_unl * weight_unl).mean()


def compute_instance_weights(model, loss_func, train_dataset, device, class_prior, lr=0.01, hop_weights=None):
    """
    Meta-gradient-based computation of per-instance weights for PU training.

    Args:
        model (nn.Module): GNN model.
        loss_func (callable): PU loss function.
        train_dataset (List[Data]): Training graphs.
        device (torch.device): Device to run on.
        class_prior (float): Estimated class prior.
        lr (float): Inner learning rate for meta-update.
        hop_weights (Tensor, optional): Per-layer weights for each graph.

    Returns:
        Tuple[Tensor, Tensor]: (instance_weights [N], val_mask [N])
    """

    model.eval()

    for i, data in enumerate(train_dataset):
        data.idx = torch.tensor(i)

    full_batch = Batch.from_data_list(train_dataset).to(device)
    train_x = full_batch.x
    train_edge_index = full_batch.edge_index
    train_batch = full_batch.batch
    train_y = full_batch.y

    with torch.no_grad():
        logits = model(train_x, train_edge_index, train_batch, hop_weights=hop_weights)
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(y=1)

    pos_idx = (train_y == 1).nonzero(as_tuple=False).view(-1)
    unl_idx = (train_y == 0).nonzero(as_tuple=False).view(-1)

    num_pos = len(pos_idx)
    neg_scores = probs[unl_idx]
    neg_sorted_idx = torch.argsort(neg_scores)
    sel_neg_idx = unl_idx[neg_sorted_idx[:num_pos]]  # confident negative

    val_idx = torch.cat([pos_idx, sel_neg_idx], dim=0)
    val_mask = torch.zeros_like(train_y, dtype=torch.bool)
    val_mask[val_idx] = True

    meta_model = deepcopy(model).to(device)
    meta_model.train()

    pred = meta_model(train_x, train_edge_index, train_batch, hop_weights=hop_weights)
    raw_losses = loss_func(pred, train_y, reduction='none')

    eps = torch.zeros(len(train_dataset), device=device, requires_grad=True)

    weighted_loss = torch.sum(raw_losses * (eps + 1e-12))

    meta_params = list(meta_model.parameters())
    grads = grad(weighted_loss, meta_params, create_graph=True)
    updated_params = [p - lr * g for p, g in zip(meta_params, grads)]

    pred_val = model.parameterized_forward(train_x, train_edge_index, train_batch, updated_params, hop_weights)
    val_loss = loss_func(pred_val[val_mask], train_y[val_mask], reduction='mean')

    grad_eps = grad(val_loss, eps, only_inputs=True)[0]

    weights = torch.clamp(-grad_eps, min=0.0)
    # weights = (1-class_prior) * (1 + weights / (weights.max() + 1e-12))

    weights = (weights / (weights.max() + 1e-12))  # normalize to [0, 1]
    weights = (1 - class_prior) + weights * class_prior  # scale to [1 - pi, 1]

    for i, data in enumerate(train_dataset):
        if data.y.item() == 1:
            weights[i] = 1.0

    return weights, val_mask

def compute_hop_weights(model, loss_func, train_dataset, device, val_mask, lr=0.01, instance_weights=None):
    """
    Compute hop-level weights via meta-gradient optimization.

    Args:
        model (nn.Module): GNN model with num_layers attribute.
        loss_func (callable): Loss function (e.g., WeightedLoss).
        train_dataset (List[Data]): Training graph list.
        device (torch.device): Device to run on.
        val_mask (Tensor): Boolean mask for meta-validation set.
        lr (float): Inner learning rate.
        instance_weights (Tensor, optional): Pre-computed instance weights.

    Returns:
        Tensor: Normalized hop weights [N_graphs x num_layers]
    """

    for i, data in enumerate(train_dataset):
        data.idx = torch.tensor(i)

    full_batch = Batch.from_data_list(train_dataset).to(device)
    x, edge_index, batch, y = full_batch.x, full_batch.edge_index, full_batch.batch, full_batch.y
    num_graphs = y.size(0)
    num_layers = model.num_layers

    meta_model = deepcopy(model).to(device)
    meta_model.train()

    hop_weights = torch.zeros((num_graphs, num_layers), device=device, requires_grad=True)

    pred = meta_model(x, edge_index, batch, hop_weights=F.softmax(hop_weights, dim=1))
    if instance_weights is not None:
        loss = loss_func(pred, y, instance_weights, reduction='mean')
    else:
        loss = loss_func(pred, y, reduction='mean')

    meta_params = list(meta_model.parameters())
    grads = grad(loss, meta_params, create_graph=True)
    updated_params = [p - lr * g for p, g in zip(meta_params, grads)]

    pred_val = model.parameterized_forward(x, edge_index, batch, updated_params)
    val_loss = loss_func(pred_val[val_mask], y[val_mask], reduction='mean')

    grad_eps = grad(val_loss, hop_weights, only_inputs=True)[0]

    weights = torch.clamp(-grad_eps*10, min=0.0)
    weights = F.softmax(weights.detach(), dim=1)

    return weights
