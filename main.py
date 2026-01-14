import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import math

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import torch.nn.functional as F

from src import *


def parse_args():
    """
    Parses command-line arguments for dataset, training, and model configuration.
    Returns:
        argparse.Namespace: Parsed arguments.
    """

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data', type=str, default='MUTAG')
    parser.add_argument('--degree-x', type=str2bool, default=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2025)

    # Experimental setting
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)
    parser.add_argument('--print-all', action='store_true', default=False)
    parser.add_argument('--observed-label-ratio', type=float, default=0.1)
    parser.add_argument('--random-drop', type=float, default=0.0)

    # Training setup
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--schedule', type=str2bool, default=True)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--meta-lr', type=float, default=1e-2)

    # Classifier
    parser.add_argument('--model', type=str, default='GIN')
    parser.add_argument('--units', type=int, default=64)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pooling', type=str, default='mean')
    parser.add_argument('--loss', type=str, default='dump')

    return parser.parse_args()


def to_device(gpu):
    """
    Returns the appropriate torch device (GPU or CPU) based on availability and user input.

    Args:
        gpu (int): GPU index to use.

    Returns:
        torch.device: Selected computation device.
    """

    if gpu is not None and torch.cuda.is_available():
        return torch.device('cuda:{}'.format(gpu))
    else:
        return torch.device('cpu')


def main():
    """
    Main training routine:
    1. Loads data and computes class prior.
    2. Initializes model, loss function, and optimizer.
    3. Trains the model using multiple iterations of PU learning.
    4. Prints evaluation results.
    """

    args = parse_args()
    device = to_device(args.gpu)
    args.seed = args.seed + args.fold
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    torch.cuda.manual_seed_all(args.seed)

    data = load_data(args.data, args.degree_x)
    prior = compute_class_prior(data)
    num_features = data.num_features
    num_classes = data.num_classes

    trn_graphs, test_graphs = load_data_fold(args.data, args.fold, args.degree_x, args.observed_label_ratio,
                                             random_drop=args.random_drop)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    model = GIN(num_features, num_classes, args.units, args.layers, args.dropout, pooling=args.pooling)
    model = model.to(device)
    loss_func = WeightedLoss(class_prior=prior)

    out_list = train_iterations(
            args, model, loss_func, test_loader, trn_graphs, prior, device, num_iterations=3,
    )

    if args.print_all:
        out = {arg: getattr(args, arg) for arg in vars(args)}
        out['all'] = out_list
        print(json.dumps(out))
    else:
        print(f'Training accuracy: {out_list["trn_acc"][-1]:.4f}')
        print(f'Test accuracy: {out_list["test_acc"][-1]:.4f}')
        print(f'Training f1: {out_list["trn_f1"][-1]:.4f}')
        print(f'Test f1: {out_list["test_f1"][-1]:.4f}')


if __name__ == '__main__':
    main()
