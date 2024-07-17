# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
import random
import math
from torch import Tensor

EPS = 1e-15


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_norm_coo = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo().todense()

    adj_torch = torch.from_numpy(adj_norm_coo).float()
    if torch.cuda.is_available():
        adj_torch = adj_torch.cuda()
    return adj_torch


def spcoo_to_torchcoo(adj):
    values = adj.data
    indices = np.vstack((adj.row ,adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    adj_torchcoo = torch.sparse_coo_tensor(i, v, adj.shape)
    return adj_torchcoo

def normalize_adj_from_tensor(adj, mode, sparse=False):
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EPS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EPS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EPS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def remove_self_loop(adjs):
    adjs_ = []
    for i in range(len(adjs)):
        adj = adjs[i].coalesce()
        diag_index = torch.nonzero(adj.indices()[0] != adj.indices()[1]).flatten()
        adj = torch.sparse.FloatTensor(adj.indices()[:, diag_index], adj.values()[diag_index], adj.shape).coalesce()
        adjs_.append(adj)
    return adjs_


def add_self_loop_and_normalize(indices, values, nnodes):
    i_indices = torch.stack((torch.tensor(range(0, nnodes)), torch.tensor(range(0, nnodes))), dim=0).to(indices.device)
    i_values = torch.ones(nnodes).to(indices.device)
    edge_indices = torch.cat((i_indices, indices), dim=1)
    edge_values = torch.cat((i_values, values),dim=0)
    edge = torch.sparse.FloatTensor(edge_indices, edge_values, ([nnodes,nnodes]))
    edge = normalize_adj_from_tensor(edge, mode='sym', sparse=True)
    return edge


def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new.coalesce()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return torch.sparse.FloatTensor(index, torch.ones(len(index[0])), adj.shape)



def find_idx(a: Tensor, b: Tensor, missing_values: int = -1):
    """Find the first index of b in a, return tensor like a."""
    a, b = a.clone(), b.clone()
    invalid = ~torch.isin(a, b)
    a[invalid] = b[0]
    sorter = torch.argsort(b)
    b_to_a: Tensor = sorter[torch.searchsorted(b, a, sorter=sorter)]
    b_to_a[invalid] = missing_values
    return b_to_a


def APPNP(h_list, adjs_o, nlayer, alpha):
    f_list = []
    for i in range(len(adjs_o)):
        h_0 = h_list[i]
        z = h_list[i]
        adj = adjs_o[i]
        for i in range(nlayer):
            z = torch.sparse.mm(adj, z)
            z = (1 - alpha) * z + alpha * h_0
        z = F.normalize(z, dim=1, p=2)
        f_list.append(z)

    return f_list


def pre_compute_dominant_view(f_list, feat):
    err_list = []
    feat = F.normalize(feat, dim=1, p=2)
    z_list = [F.normalize(z, dim=1, p=2) for z in f_list]
    feat_sim = torch.mm(feat, feat.t())
    for i in range(len(z_list)):
        embed = z_list[i]
        z_sim = torch.mm(embed, embed.t())
        err = F.mse_loss(z_sim, feat_sim)
        err_list.append(err)

    dominant_index = torch.argmin(torch.tensor(err_list))
    return dominant_index

def graph_process(adjs, feat, args):
    adjs = [adj.coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)  # return sparse tensor
    adj_I = torch.eye(len(feat)).to(feat.device)

    adjs_o = [normalize_adj_from_tensor(adj_I+adj.to_dense(), mode='sym').to_sparse() for adj in adjs]

    print(adjs_o)

    return adjs_o

def graph_process_large(adjs, feat, args):
    adjs = [adj.coalesce() for adj in adjs]
    adjs = remove_self_loop(adjs)  # return sparse tensor
    adjs = [sparse_tensor_add_self_loop(adj) for adj in adjs]

    adjs_o = [normalize_adj_from_tensor(adj, mode='sym', sparse=True) for adj in adjs]

    print(adjs_o)

    return adjs_o

def pre_compute_dominant_view_large(f_list, feat):
    err_list = []
    feat = F.normalize(feat, dim=1, p=2)
    z_list = [F.normalize(z, dim=1, p=2) for z in f_list]

    nnodes = len(feat)
    batchsize = 2500
    batchnum = math.ceil(nnodes / batchsize)
    for i in range(len(z_list)):
        embed = z_list[i]
        err = 0
        for batch in range(batchnum):
            start_index = batch * batchsize
            end_index = start_index + batchsize
            if end_index > nnodes:
                end_index = nnodes
                start_index = end_index - batchsize

            z_sim = torch.mm(embed[start_index:end_index], embed[start_index:end_index].t())
            feat_sim = torch.mm(feat[start_index:end_index], feat[start_index:end_index].t())

            err += F.mse_loss(z_sim, feat_sim)

        err_list.append(err / batchnum)

    dominant_index = torch.argmin(torch.tensor(err_list))
    return dominant_index
