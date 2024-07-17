import numpy as np
import torch
from utils import load_data, set_params, clustering_metrics
from module.BMGC import *
from module.preprocess import *
import warnings
import datetime
import random
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
args = set_params()

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

    adjs_o = graph_process(adjs, feat, args)

    f_list = APPNP([feat for _ in range(sub_num)], adjs_o, args.nlayer, args.filter_alpha)
    dominant_index = pre_compute_dominant_view(f_list, feat)

    model = BMGC(feats_dim, sub_num, args.hidden_dim, args.embed_dim, nb_classes, args.tau, args.dropout, len(feat), dominant_index, args.nlayer, device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.l2_coef)

    if torch.cuda.is_available():
        model.cuda()

    period = 50

    fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
    print(args, file=fh)
    fh.write('\r\n')
    fh.flush()
    fh.close()

    if args.load_parameters == False:

        for epoch in range(args.nb_epochs):
            flag = False
            model.train()
            optimizer.zero_grad()
            if (epoch+1) % period == 0:
                flag = True

            loss = model(feat, f_list, flag)

            loss.backward()
            optimizer.step()

            print("Epoch:", epoch)
            print('Total loss: ', loss.item())

            if (epoch + 1) % period == 0:
                model.eval()
                embeds = model.get_embeds(f_list).cpu().numpy()

                estimator = KMeans(n_clusters=nb_classes)
                ACC_list = []
                F1_list = []
                NMI_list = []
                ARI_list = []
                for _ in range(10):
                    estimator.fit(embeds)
                    y_pred = estimator.predict(embeds)
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

                print(
                    '\t[Clustering] ACC: {:.2f}   F1: {:.2f}  NMI: {:.2f}   ARI: {:.2f} \n'.format(np.round(acc * 100, 2),
                                                                                                np.round(f1 * 100, 2),
                                                                                                np.round(nmi * 100, 2),
                                                                                                np.round(ari * 100, 2)))
                fh = open("result_" + args.dataset + "_NMI&ARI.txt", "a")
                fh.write(
                    'ACC=%f, f1_macro=%f,  NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1, nmi, ari))
                fh.write('\r\n')
                fh.flush()
                fh.close()

        torch.save(model.state_dict(), './checkpoint/' + args.dataset + '/best_' + str(args.seed) + '.pth')

    else:
        model.load_state_dict(torch.load('./best/' + args.dataset + '/best_' + str(0) + '.pth'))

    model.cuda()
    print("---------------------------------------------------")
    model.eval()
    embeds = model.get_embeds(f_list).cpu().numpy()

    estimator = KMeans(n_clusters=nb_classes)
    ACC_list = []
    F1_list = []
    NMI_list = []
    ARI_list = []
    for _ in range(10):
        estimator.fit(embeds)
        y_pred = estimator.predict(embeds)
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


if __name__ == '__main__':

        set_seed(args.seed)
        train()

