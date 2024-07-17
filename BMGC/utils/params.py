import argparse
import torch
import sys

# argv = sys.argv
# dataset = argv[1]
dataset = 'acm-3025'

def acm_3025_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="acm-3025")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nlayer', type=int, default=3)

    # The parameters of learning process
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    return args


def acm_4019_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="acm-4019")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--nlayer', type=int, default=3)

    # The parameters of learning process
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.3)

    args, _ = parser.parse_known_args()
    return args


def dblp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="dblp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nlayer', type=int, default=3)

    # The parameters of learning process
    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.)

    args, _ = parser.parse_known_args()
    return args


def amazon_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="amazon")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--nlayer', type=int, default=3)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.5)

    args, _ = parser.parse_known_args()
    return args


def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="yelp")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--nlayer', type=int, default=3)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.3)

    args, _ = parser.parse_known_args()
    return args


def csbm_20_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="csbm-20")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--nlayer', type=int, default=3)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.)

    args, _ = parser.parse_known_args()
    return args


def csbm_50_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="csbm-50")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--nlayer', type=int, default=3)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.)

    args, _ = parser.parse_known_args()
    return args


def csbm_100_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="csbm-100")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--nlayer', type=int, default=3)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.)

    args, _ = parser.parse_known_args()
    return args


def csbm_150_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="csbm-150")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=10)
    parser.add_argument('--nb_epochs', type=int, default=400)
    parser.add_argument('--nlayer', type=int, default=3)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.)

    args, _ = parser.parse_known_args()
    return args


def mag_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_parameters', default=True)
    parser.add_argument('--dataset', type=str, default="mag")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--nb_epochs', type=int, default=8)
    parser.add_argument('--nlayer', type=int, default=2)

    parser.add_argument('--l2_coef', type=float, default=1e-4)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--tau', type=float, default=1.0)

    # model-specific parameters
    parser.add_argument('--filter_alpha', type=float, default=0.)
    args, _ = parser.parse_known_args()
    return args


def set_params():
    if dataset == "acm-3025":
        args = acm_3025_params()
    elif dataset == "acm-4019":
        args = acm_4019_params()
    elif dataset == "dblp":
        args = dblp_params()
    elif dataset == 'amazon':
        args = amazon_params()
    elif dataset == 'yelp':
        args = yelp_params()
    elif dataset == 'csbm-20':
        args = csbm_20_params()
    elif dataset == 'csbm-50':
        args = csbm_50_params()
    elif dataset == 'csbm-100':
        args = csbm_100_params()
    elif dataset == 'csbm-150':
        args = csbm_150_params()

    return args


def set_params_large():
    if dataset == 'mag':
        args = mag_params()
    return args