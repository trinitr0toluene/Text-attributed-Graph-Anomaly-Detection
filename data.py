from __future__ import division
from torch.utils.data import Dataset
import numpy as np
import torch
import random


def pad_2d_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


class Batch():
    def __init__(self, attn_bias, x, ids, s_n, t_n):
        super(Batch, self).__init__()
        self.x = x
        self.attn_bias = attn_bias
        self.ids = ids
        self.s_n = s_n
        self.t_n = t_n

    def to(self, device):
        self.s_n = self.s_n.to(device)
        self.t_n = self.t_n.to(device)
        self.x = self.x.to(device)
        self.attn_bias = self.attn_bias.to(device)
        self.ids = self.ids.to(device)
        return self

    def __len__(self):
        return self.s_n.size(0)


def collator(items, feature, shuffle=False, perturb=False):
    batch_list = []
    s_n_list = []
    t_n_list = []
    for item in items:
        s_n_list.append(torch.tensor([item[0]]))
        t_n_list.append(torch.from_numpy(item[1]))
        for data in item[2]:
            attn_bias = data[0]
            feature_id = data[1]
            batch_list.append((attn_bias, feature_id))
    if shuffle:
        random.shuffle(batch_list)
    s_n = torch.cat(s_n_list)
    t_n = torch.cat(t_n_list)
    attn_biases, xs = zip(*batch_list)
    max_node_num = max(i.size(0) for i in xs)
    x = torch.cat([pad_2d_unsqueeze(feature[i], max_node_num) for i in xs])
    ids = torch.cat([i.unsqueeze(0) for i in xs])
    if perturb:
        x += torch.FloatTensor(x.shape).uniform_(-0.1, 0.1)
    attn_bias = torch.cat([i.unsqueeze(0) for i in attn_biases])

    return Batch(attn_bias=attn_bias, x=x, ids=ids, s_n=s_n, t_n=t_n)


class DataHelper(Dataset):
    def __init__(self, edge_index, args, directed=False, transform=None):
        # self.num_nodes = len(node_list)
        self.transform = transform
        self.degrees = dict()
        self.node_set = set()
        self.neighs = dict()
        self.args = args

        idx, degree = np.unique(edge_index, return_counts=True)
        for i in range(idx.shape[0]):
            self.degrees[idx[i]] = degree[i].item()

        self.node_dim = idx.shape[0]
        print('lenth of dataset', self.node_dim)

        train_edge_index = edge_index
        self.final_edge_index = train_edge_index.T

        for i in range(self.final_edge_index.shape[0]):
            s_node = self.final_edge_index[i][0].item()
            t_node = self.final_edge_index[i][1].item()

            if s_node not in self.neighs:
                self.neighs[s_node] = []
            if t_node not in self.neighs:
                self.neighs[t_node] = []

            self.neighs[s_node].append(t_node)
            if not directed:
                self.neighs[t_node].append(s_node)

        # self.neighs = sorted(self.neighs)
        self.idx = idx

    def __len__(self):
        return self.node_dim

    def __getitem__(self, idx):
        s_n = self.idx[idx].item()
        t_n = [np.random.choice(self.neighs[s_n], replace=True).item() for _ in range(self.args.neigh_num)]
        t_n = np.array(t_n)

        sample = {
            's_n': s_n,  # e.g., 5424
            't_n': t_n,  # e.g., 5427
            # 'neg_n': neg_n
        }

        if self.transform:
            sample = self.transform(sample)

        return sample





