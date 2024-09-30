import torch
import numpy as np
import scipy.sparse as sp

from tqdm import tqdm
from sklearn import preprocessing


def adj_normalize(mx):
    "A' = (D + I)^-1/2 * ( A + I ) * (D + I)^-1/2"
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def column_normalize(mx):
    "A' = A * D^-1 "
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = mx.dot(r_mat_inv)
    return mx


def process_data(dataset):
    num_nodes = 0
    tit_list = []
    with open('data/' + dataset + '/train_text.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            tit_list.append(line[2])
            num_nodes += 1
    f.close()

    print('num_nodes', num_nodes)

    tit_list_arr = np.array(tit_list)
    np.save('dataset/' + dataset + '/text', tit_list_arr)

    raw_edge_index = [[], []]
    with open('data/'+dataset+'/mapped_edges.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split()
            raw_edge_index[0].append(int(line[0]))
            raw_edge_index[1].append(int(line[1]))
    f.close()

    edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
    edge_index = np.array(edge_index)
    print(edge_index.shape)

    adj = sp.coo_matrix((np.ones(edge_index.shape[1]), (edge_index[0], edge_index[1])),
                        shape=(num_nodes, num_nodes), dtype=np.float32)
    normalized_adj = adj_normalize(adj)
    column_normalized_adj = column_normalize(adj)

    c = 0.15
    k1 = 15
    num_subgraph = 8

    power_adj_list = [normalized_adj]
    for m in range(5):
        power_adj_list.append(power_adj_list[0] * power_adj_list[m])

    # feature = np.load('data/cora/node_f.npy')
    # feature = preprocessing.StandardScaler().fit_transform(feature)
    sampling_matrix = c * np.linalg.inv((sp.eye(adj.shape[0]) - (1 - c) * normalized_adj).toarray())

    # create sub_graph for node
    data_list = []
    for id in tqdm(range(num_nodes)):
        s = sampling_matrix[id]
        s[id] = -1000.0
        top_neighbor_index = s.argsort()[-k1:]

        s = sampling_matrix[id]
        s[id] = 0
        s = np.maximum(s, 0)
        sample_num1 = np.minimum(k1, (s > 0).sum())

        sub_data_list = []
        for _ in range(num_subgraph):
            if sample_num1 > 0:
                sample_index1 = np.random.choice(a=np.arange(num_nodes), size=sample_num1, replace=False,
                                                 p=s / s.sum())
            else:
                sample_index1 = np.array([], dtype=int)

            node_feature_id = torch.cat([torch.tensor([id, ]), torch.tensor(sample_index1, dtype=int),
                                         torch.tensor(top_neighbor_index[: k1 - sample_num1], dtype=int)])
            # create attention bias (positional encoding)
            attn_bias = torch.cat(
                [torch.tensor(i[node_feature_id, :][:, node_feature_id].toarray(), dtype=torch.float32).unsqueeze(0) for
                 i in power_adj_list])
            attn_bias = attn_bias.permute(1, 2, 0)

            feature_id = node_feature_id
            assert len(feature_id) == k1 + 1
            sub_data_list.append([attn_bias, feature_id])
        data_list.append(sub_data_list)
    torch.save(data_list, './dataset/' + dataset + '/subgraph.pt')


if __name__ == '__main__':
    process_data("cora")