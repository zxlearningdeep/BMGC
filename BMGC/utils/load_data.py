import numpy as np
import scipy.sparse as sp
import torch
import torch as th
from sklearn.preprocessing import OneHotEncoder
import scipy.io as sio
from module.preprocess import remove_self_loop, find_idx
import pickle as pkl
import torch.nn.functional as F

def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def adj_values_one(adj):
    adj = adj.coalesce()
    index = adj.indices()
    return th.sparse.FloatTensor(index, th.ones(len(index[0])), adj.shape)

def sparse_tensor_add_self_loop(adj):
    adj = adj.coalesce()
    node_num = adj.shape[0]
    index = torch.stack((torch.tensor(range(node_num)), torch.tensor(range(node_num))), dim=0).to(adj.device)
    values = torch.ones(node_num).to(adj.device)

    adj_new = torch.sparse.FloatTensor(torch.cat((index, adj.indices()), dim=1), torch.cat((values, adj.values()),dim=0), adj.shape)
    return adj_new


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def sp_tensor_to_sp_csr(adj):
    adj = adj.coalesce()
    row = adj.indices()[0]
    col = adj.indices()[1]
    data = adj.values()
    shape = adj.size()
    adj = sp.csr_matrix((data, (row, col)), shape=shape)
    return adj



def load_acm_4019():

    path = "./data/acm-4019/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)

    feat_p = sp.load_npz(path + "p_feat.npz")

    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    adjs = [pap, psp]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj).coalesce() for adj in adjs]

    label = th.FloatTensor(label)

    feat_p = th.FloatTensor(preprocess_features(feat_p))

    return feat_p, adjs, label


def load_acm_3025():
    """load dataset ACM

    Returns:
        gnd(ndarray): [nnodes,]
    """

    # Load data
    dataset = "./data/acm-3025/" + 'ACM3025'
    data = sio.loadmat('{}.mat'.format(dataset))
    X = data['feature']
    A = data['PAP']
    B = data['PLP']

    if sp.issparse(X):
        X = X.todense()

    A = np.array(A)
    B = np.array(B)
    X = np.array(X)


    Adj = []
    Adj.append(A)
    Adj.append(B)

    gnd = data['label']
    gnd = gnd.T
    gnd = np.argmax(gnd, axis=0)

    X = torch.tensor(X).float()
    Adj = [torch.tensor(adj).to_sparse() for adj in Adj]
    Adj = remove_self_loop(Adj)
    label = encode_onehot(gnd)
    label = th.FloatTensor(label)

    X = F.normalize(X, dim=1, p=2)

    return X, Adj, label


def load_dblp():
    path = "./data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    feat_a = sp.load_npz(path + "a_feat.npz").astype("float32")

    apa = sp.load_npz(path + "apa.npz")  
    apcpa = sp.load_npz(path + "apcpa.npz")
    aptpa = sp.load_npz(path + "aptpa.npz")
    adjs = [apa, apcpa, aptpa]
    adjs = [sparse_mx_to_torch_sparse_tensor(adj).coalesce() for adj in adjs]
    
    label = th.FloatTensor(label)
    feat_a = th.FloatTensor(preprocess_features(feat_a))

    return feat_a, adjs, label


def load_amazon():
    data = pkl.load(open("data/amazon/amazon.pkl", "rb"))
    label = data['label'].argmax(1)
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    # dense
    ivi = torch.from_numpy(data["IVI"]).float()
    ibi = torch.from_numpy(data["IBI"]).float()
    ioi = torch.from_numpy(data["IOI"]).float()
    adj = []
    adj.append(ivi)
    adj.append(ibi)
    adj.append(ioi)
    adj = [a.to_sparse().coalesce() for a in adj]

    X = torch.from_numpy(data['feature']).float()
    rowsum = X.sum(dim=1)
    r_inv = th.pow(rowsum, -1).flatten()
    r_inv[th.isinf(r_inv)] = 0.
    r_inv = r_inv.view(-1,1)
    X = X * r_inv
    return X, adj, label


def load_yelp():

    path = "./data/yelp/"
    feat_b = sp.load_npz(path + "features_0.npz").astype("float32")
    feat_b = th.FloatTensor(preprocess_features(feat_b))
    label = np.load(path+'labels.npy')
    label = encode_onehot(label)
    label = th.FloatTensor(label)

    blb = np.load(path+'blb.npy').astype("float32")
    bsb = np.load(path+'bsb.npy').astype("float32")
    bub = np.load(path+'bub.npy').astype("float32")


    adjs = [bsb, bub, blb]
    adjs = [th.tensor(adj).to_sparse() for adj in adjs]


    return feat_b, adjs, label



def load_csbm_20():
    r = 20

    path = "./data/csbm/"
    label = th.load(path+'label.pt')
    label = F.one_hot(label, num_classes=2)

    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'adj_v_0.pt').coalesce()
    adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
    adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
    adjs = [adj_1, adj_2, adj_3]

    return feat, adjs, label



def load_csbm_50():
    r = 50

    path = "./data/csbm/"
    label = th.load(path+'label.pt')
    label = F.one_hot(label, num_classes=2)

    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'adj_v_0.pt').coalesce()
    adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
    adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
    adjs = [adj_1, adj_2, adj_3]

    return feat, adjs, label


def load_csbm_100():
    r = 100

    path = "./data/csbm/"
    label = th.load(path+'label.pt')
    label = F.one_hot(label, num_classes=2)

    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'adj_v_0.pt').coalesce()
    adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
    adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
    adjs = [adj_1, adj_2, adj_3]

    return feat, adjs, label



def load_csbm_150():
    r = 150

    path = "./data/csbm/"
    label = th.load(path+'label.pt')
    label = F.one_hot(label, num_classes=2)

    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'adj_v_0.pt').coalesce()
    adj_2 = th.load(path+str(r)+'/adj_v_1.pt').coalesce()
    adj_3 = th.load(path+str(r)+'/adj_v_2.pt').coalesce()
    adjs = [adj_1, adj_2, adj_3]

    return feat, adjs, label

def load_mag():

    path = "./data/mag-4/"
    label = th.load(path+'label.pt')
    label = F.one_hot(label, num_classes=4)

    feat = th.load(path+'feat.pt').float()
    adj_1 = th.load(path+'pap.pt').coalesce()
    adj_2 = th.load(path+'pp.pt').coalesce()
    adjs = [adj_1, adj_2]

    return feat, adjs, label


def load_data(dataset):
    if dataset == "acm-3025":
        data = load_acm_3025()
    elif dataset == "acm-4019":
        data = load_acm_4019()
    elif dataset == "dblp":
        data = load_dblp()
    elif dataset == 'amazon':
        data = load_amazon()
    elif dataset == 'yelp':
        data = load_yelp()
    elif dataset == 'csbm-20':
        data = load_csbm_20()
    elif dataset == 'csbm-50':
        data = load_csbm_50()
    elif dataset == 'csbm-100':
        data = load_csbm_100()
    elif dataset == 'csbm-150':
        data = load_csbm_150()
    elif dataset == 'mag':
        data = load_mag()

    return data



