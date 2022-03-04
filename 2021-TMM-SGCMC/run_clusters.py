import argparse
import os
import tensorflow as tf
from Network.Trainer import Trainer
import process
import numpy as np
import scipy.io as sio
from utils import load_data, load_graph
import random


def parse_args():
    """
    Parses the arguments.
    """
    parser = argparse.ArgumentParser(description="Run gate.")
    parser.add_argument('--dataset', nargs='?', default='cora', help='Input dataset')
    parser.add_argument('--seed', type=int, default=30) #33
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate. Default is 0.001.')
    parser.add_argument('--n-epochs', default=30, type=int, help='Number of epochs')
    parser.add_argument('--hidden-dims-1', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--hidden-dims-2', type=list, nargs='+', default=[512, 512], help='Number of dimensions1.')
    parser.add_argument('--lambda-', default=1.0, type=float, help='Parameter controlling the contribution of edge '
                                                                   'reconstruction in the loss function.')
    parser.add_argument('--dropout', default=0.0, type=float, help='Dropout.')
    parser.add_argument('--gradient_clipping', default=3.0, type=float, help='gradient clipping')
    return parser.parse_args()


def main(args):
    """
    Load HARR Dataset.
    """
    dataset = load_data('hhar')
    G = load_graph('hhar', 5)
    X = dataset.x

    Label = dataset.y
    # prepare the data
    G_tf, S, R = process.prepare_graph_data(G)
    G_tf2 = G_tf
    X2 = sio.loadmat('X2' + '.mat')
    X2_dict = dict(X2)
    X2 = X2_dict['X2']
    S2 = S
    R2 = R

    # add feature dimension size to the beginning of hidden_dims
    feature_dim1 = X.shape[1]
    args.hidden_dims_1 = [feature_dim1] + args.hidden_dims_1
    feature_dim2 = X2.shape[1]
    args.hidden_dims_2 = [feature_dim2] + args.hidden_dims_2
    print('Dim_hidden_1: ' + str(args.hidden_dims_1))
    print('Dim_hidden_2: ' + str(args.hidden_dims_2))
    trainer = Trainer(args)
    trainer(G_tf, G_tf2, X, X2, S, S2, R, R2, Label)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    my_tf_seed = args.seed
    setup_seed(my_tf_seed)
    main(args)
