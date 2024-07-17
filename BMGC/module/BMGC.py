# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .preprocess import *
from .loss_fun import *


class BMGC(nn.Module):
    def __init__(self, feats_dim, sub_num, hidden_dim, embed_dim, num_clusters, tau, dropout, nnodes, dominant_index, nlayer, device):
        super(BMGC, self).__init__()
        self.feats_dim = feats_dim
        self.embed_dim = embed_dim
        self.sub_num = sub_num
        self.tau = tau
        self.device = device
        self.nlayer = nlayer
        self.num_clusters = num_clusters
        self.dominant_index = dominant_index

        self.fc = nn.Sequential(nn.Dropout(dropout),
                                nn.Linear(feats_dim, hidden_dim),
                                nn.ELU(),
                                nn.Dropout(dropout),
                                nn.Linear(hidden_dim, embed_dim),
                                )

        self.decoder = nn.Sequential(nn.Dropout(dropout),
                                     nn.Linear(embed_dim, hidden_dim),
                                     nn.ELU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_dim, feats_dim))
        self.proj = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ELU(),
            nn.Linear(embed_dim, embed_dim)
        ) for _ in range(self.sub_num)])

        self.contrast = Contrast(self.tau)

    def forward(self, feat, f_list, re_computing=False):

        z_list = self.encoder(f_list)
        dominant_index, err_loss = self.dominant_view_mining(z_list, feat)
        if re_computing:
            self.dominant_index = dominant_index

        ae_loss = self.ae_loss(z_list, f_list)

        mi_loss = self.mi_loss(z_list, self.dominant_index)
        clu_loss = KL_clustering(self.device, z_list, self.num_clusters, self.dominant_index)
        loss = ae_loss + mi_loss + err_loss + clu_loss

        return loss

    def encoder(self, x_list):
        z_list = []

        for i in range(len(x_list)):
            z = self.fc(x_list[i])
            z_list.append(z)

        return z_list

    def ae_loss(self, z_list, f_list):
        loss_rec = 0
        for i in range(self.sub_num):
            fea_rec = self.decoder(z_list[i])
            loss_rec += sce_loss(fea_rec, f_list[i])
        loss_ae = loss_rec / self.sub_num
        return loss_ae

    def mi_loss(self, z_list, dominant_index):
        pos = torch.eye(len(z_list[0])).to(self.device).to_sparse()
        proj_list = [self.proj[i](z_list[i]) for i in range(self.sub_num)]
        loss = 0
        for i in range(self.sub_num):
            if i != dominant_index:
                loss += self.contrast.compute_loss(proj_list[dominant_index], proj_list[i], pos)

        loss = loss / (self.sub_num - 1)

        return loss

    def dominant_view_mining(self, z_list, feat):

        err_list = []
        feat = F.normalize(feat, dim=1, p=2)
        z_list = [F.normalize(z, dim=1, p=2) for z in z_list]
        feat_sim = torch.mm(feat, feat.t())
        for i in range(len(z_list)):
            embed = z_list[i]
            z_sim = torch.mm(embed, embed.t())
            err = F.mse_loss(z_sim, feat_sim)
            err_list.append(err)

        dominant_index = torch.argmin(torch.tensor(err_list))
        err_loss = sum(err_list) / len(err_list)

        return dominant_index, err_loss


    def get_embeds(self, f_list):

        z_list = self.encoder(f_list)
        z = torch.cat(z_list, dim=1)
        z = F.normalize(z, dim=1, p=2)
        return z.detach()
