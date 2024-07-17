import numpy as np
import torch
from utils import load_data, set_params_large, clustering_metrics
from module.BMGC import *
from module.preprocess import *
import warnings
import datetime
import time
import random
from kmeans_pytorch import kmeans
from torch.utils.data import RandomSampler

warnings.filterwarnings('ignore')
args = set_params_large()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train():
    feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    num_target_node = len(feat)

    feats_dim = feat.shape[1]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print(args)

    if torch.cuda.is_available():
        print('Using CUDA')
        adjs = [adj.cuda() for adj in adjs]
        feat = feat.cuda()

    adjs_o = graph_process_large(adjs, feat, args)

    f_list = APPNP([feat for _ in range(sub_num)], adjs_o, args.nlayer, args.filter_alpha)
    dominant_index = pre_compute_dominant_view_large(f_list, feat)

    acc_ss = []
    F1_ss = []
    nmi_ss = []
    ari_ss = []
    print("Started training...")
    for ss in [0, 1, 2, 3, 4]:
        seed = ss
        set_seed(seed)

        model = BMGC(feats_dim, sub_num, args.hidden_dim, args.embed_dim, nb_classes, args.tau, args.dropout, len(feat), dominant_index, args.nlayer, args.gamma, args.mu, device)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_coef)

        if torch.cuda.is_available():
            model.cuda()

        period = 2
        batchsize = 5000
        batch_num = math.ceil(num_target_node / batchsize)

        starttime = datetime.datetime.now()

        fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
        print(args, file=fh)
        fh.write('\r\n')
        fh.flush()
        fh.close()

        if args.load_parameters == False:
            t0 = time.time()

            for epoch in range(args.nb_epochs):
                model.train()
                sampler_ = RandomSampler(range(num_target_node), replacement=False)
                sampler = torch.tensor([i for i in sampler_]).to(device)

                for batch_index in range(batch_num):
                    start_index = batch_index * batchsize
                    end_index = start_index + batchsize
                    if end_index > num_target_node:
                        end_index = num_target_node
                        start_index = end_index - batchsize
                    seed_node = sampler[start_index:end_index]

                    feat_sampler = feat[seed_node]
                    f_list_sampler = [f[seed_node] for f in f_list]

                    optimizer.zero_grad()

                    loss = model(feat_sampler, f_list_sampler, False)
                    loss.backward()
                    optimizer.step()


            t1 = time.time()
            training_time = t1 - t0
            training_time = format_time(training_time)
            print("Training Time:", training_time)
            torch.save(model.state_dict(), './checkpoint/' + args.dataset + '/best_' + str(seed) + '.pth')

        else:
            model.load_state_dict(torch.load('./best/' + args.dataset + '/best_' + str(0) + '.pth'))

        model.cuda()
        print("---------------------------------------------------")
        model.eval()
        embeds = model.get_embeds(f_list)

        ACC_list = []
        F1_list = []
        NMI_list = []
        ARI_list = []
        for _ in range(10):

            y_pred, _ = kmeans(X=embeds, num_clusters=nb_classes, distance='euclidean',
                                     device=device)
            y_pred = y_pred.cpu().numpy()
            cm = clustering_metrics(torch.argmax(label, dim=-1).numpy(), y_pred, args.dataset)
            ac, nm, f1, ari = cm.evaluationClusterModelFromLabel()

            ACC_list.append(ac)
            F1_list.append(f1)
            NMI_list.append(nm)
            ARI_list.append(ari)
        acc = sum(ACC_list) / len(ACC_list)
        f1 = sum(F1_list) / len(F1_list)
        ari = sum(ARI_list) / len(ARI_list)
        nmi = sum(NMI_list) / len(NMI_list)

        print('\t[Clustering] ACC: {:.2f}   F1: {:.2f}  NMI: {:.2f}   ARI: {:.2f} \n'.format(np.round(acc*100,2), np.round(f1*100,2),
                                                                                          np.round(nmi*100,2), np.round(ari*100,2)))
        fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
        fh.write(
            'ACC=%f, f1_macro=%f,  NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1, nmi, ari))
        fh.write('\r\n')
        fh.write('---------------------------------------------------------------------------------------------------')
        fh.write('\r\n')
        fh.flush()
        fh.close()

        acc_ss.append(acc)
        F1_ss.append(f1)
        ari_ss.append(ari)
        nmi_ss.append(nmi)
    print(
        '\t[Clustering]  ACC_mean: {:.2f} var: {:.2f}  F1_mean: {:.2f} var: {:.2f}  NMI_mean: {:.2f} var: {:.2f}  ARI_mean: {:.2f} var: {:.2f} \n'.format(
            np.mean(acc_ss) * 100,
            np.std(acc_ss) * 100,
            np.mean(
                F1_ss) * 100,
            np.std(
                F1_ss) * 100,
            np.mean(
                nmi_ss) * 100,
            np.std(
                nmi_ss) * 100,
            np.mean(
                ari_ss) * 100,
            np.std(
                ari_ss) * 100, ))


if __name__ == '__main__':

        train()

