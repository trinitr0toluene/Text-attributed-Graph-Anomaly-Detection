from torch.utils.data import Subset
from sklearn import preprocessing
import numpy as np
import argparse
import torch
# import dgl
from random import sample
import random
import math
import time
from torch_geometric.data import Dataset, Data, Batch
from torch_geometric.loader import DataLoader
from model import CLIP, tokenize
from data import DataHelper, collator
from functools import partial



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True



def main(args):
    setup_seed(seed)

    model = CLIP(args).to(device)
    # print(model)

    model.train()

    for j in range(args.epoch_num):
        loss = 0
        for i_batch, graph in enumerate(train_loader):
            # optimizer.zero_grad()
            graph = graph.to(device)
            # x = graph.x
            # edge_index = graph.edge_index
            # root_features = graph.root_features

            # batched_graph = batched_graph.to(device)
            # root_features =root_features.to(device)
            # s_n, t_n = batched_graph['s_n'], batched_graph['t_n']
            # s_n_arr, t_n_arr = s_n.numpy(), t_n.numpy().reshape(-1)
            # # .reshape((1, -1))
            # s_n_text, t_n_text = np.array(tit_list)[s_n_arr].tolist(), np.array(tit_list)[t_n_arr].tolist()
            # s_n_text, t_n_text = tokenize(s_n_text, context_length=args.context_length).to(device), tokenize(t_n_text, context_length=args.context_length).to(device)
            # root_f = data.root_features
            
            running_loss = model.forward(graph, device)
            loss += running_loss
            # if i_batch >2 :
            #     break
            if j == 0 and i_batch % 100 == 0:
                print('{}th loss in the first epoch:{}'.format(i_batch, running_loss))

        # break
        loss = loss/args.batch_size
        print('{}th epoch loss:{}'.format(j, loss))

        # optimizer.zero_grad()  
        loss.backward()
        # optimizer.step()

    torch.save(model.state_dict(), './res/{}/node_ttgt_8&12_0.1_pt.pkl'.format(args.data_name))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--epoch_num', type=int, default=3, help='epoch number')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--edge_coef', type=float, default=10)
    parser.add_argument('--neigh_num', type=int, default=3)

    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--gnn_layers', type=int, default=2)
    parser.add_argument('--trans_layers', type=int, default=1)
    parser.add_argument('--gnn_input', type=int, default=768)
    parser.add_argument('--gnn_hid', type=int, default=128)
    parser.add_argument('--gnn_output', type=int, default=128)
    parser.add_argument('--gnn_dropout', type=float, default=0.5)
    parser.add_argument('--trans_dropout', type=float, default=0.2)
    parser.add_argument('--graph_weight', type=float, default=0.8)
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='weight for residual link')
    parser.add_argument('--use_bn', action='store_true', help='use layernorm')
    parser.add_argument('--use_residual', action='store_true',
                        help='use residual link for each GNN layer')
    parser.add_argument('--use_weight', action='store_true',
                        help='use weight for GNN convolution')
    parser.add_argument('--use_init', action='store_true', help='use initial feat for each GNN layer')
    parser.add_argument('--use_act', action='store_true', help='use activation for each GNN layer')
    parser.add_argument('--aggregate', type=str, default='add',
                        help='aggregate type, add or cat.')

    parser.add_argument('--context_length', type=int, default=128)

    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--transformer_heads', type=int, default=8)
    parser.add_argument('--transformer_layers', type=int, default=12)
    parser.add_argument('--transformer_width', type=int, default=768)
    parser.add_argument('--vocab_size', type=int, default=49408) # 49408
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_name', type=str, default='cora')

    args = parser.parse_args()

node_file = './data/FakeNews/gossipcop/new_bert_feature.npz'
graph_id_file = './data/FakeNews/gossipcop/node_graph_id.npy'
adj_file = './data/FakeNews/gossipcop/A.txt'
labels_file = './data/FakeNews/gossipcop/graph_labels.npy'
train_file = './data/FakeNews/gossipcop/train_idx.npy'
val_file  = './data/FakeNews/gossipcop/val_idx.npy'
test_file = './data/FakeNews/gossipcop/test_idx.npy'




def find_root(file):

    node_graph_id = np.load(file)
    root = {}
    root_list = []

    for idx, graph_id in enumerate(node_graph_id):
        if graph_id not in root:
            root[graph_id] = idx
            root_list.append(idx)

    return root_list





class FakeNewsDataset(Dataset):
    def __init__(self, node_file, graph_id_file, adj_file,train_file, test_file, val_file):

        self.node_graph_id = np.load(graph_id_file)
        self.root_list = find_root(graph_id_file) ## root_node_idx
        self.graph_ids = np.unique(self.node_graph_id) ## graph_id:(0, num_graphs-1)
        
        self.node_f = np.load(node_file)['data'].reshape((314262, 768))
        # self.node_f = preprocessing.StandardScaler().fit_transform(node_f)
        
        self.edges = []
        with open(adj_file, 'r') as f:
            for line in f:
                src, dst = map(int, line.strip().split(','))
                self.edges.append((src, dst))
        self.edges = np.array(self.edges).T
             
        self.graph_labels = np.load(labels_file)

        self.train_idx = np.load(train_file)
        self.test_idx = np.load(test_file)
        self.val_idx = np.load(val_file)
        
    def __len__(self):
        return len(self.graph_ids)
    
    def __getitem__(self,idx):

        ### idx = graph id
        # graph_id = self.graph_ids[idx]

        node_mask = self.node_graph_id == idx
        graph_node_ids = np.where(node_mask)[0]
        node_features = self.node_f[node_mask]

        root_idx = self.root_list[idx]
        root_features = self.node_f[root_idx]
        
        edge_mask = np.logical_and(node_mask[self.edges[0]], node_mask[self.edges[1]])
        # graph_edges = self.edges[:, (node_mask[self.edges[0]] & node_mask[self.edges[1]])]
        graph_edges = self.edges[:,edge_mask]

        mapping = {global_id: local_id for local_id, global_id in enumerate(graph_node_ids)}   
        graph_edges = np.vectorize(mapping.get)(graph_edges)

        batch = torch.zeros(len(graph_node_ids), dtype=torch.long)
        # graph_edges = np.array([[graph_node_ids.tolist().index(e[0]), graph_node_ids.tolist().index(e[1])] for e in graph_edges.T]).T

        # graph = dgl.graph((graph_edges[0], graph_edges[1]))
        # graph.ndata['feat'] = torch.tensor(node_features, dtype=torch.float32)
        graph_label = self.graph_labels[idx]
        data = Data(x=torch.tensor(node_features, dtype=torch.float),
                    edge_index=torch.tensor(graph_edges, dtype=torch.long),
                    batch=batch,
                    root_features=torch.tensor(root_features, dtype=torch.float),
                    label = torch.tensor(graph_label,dtype = torch.float)
                    )

        return data

        
# def collate_fn(batch):
#     # graphs, root = zip(*batch)
    
#     # batched_graph = dgl.batch(graphs)
#     # # labels = torch.tensor(labels, dtype=torch.long)
#     # root_features = torch.tensor(root_features, dtype=torch.float32)
#     graphs = Batch.from_data_list(batch)
#     root_f = torch.stack([graph.root_features for graph in batch])
#     graphs.root_features = root_f
    
#     return graphs

dataset = FakeNewsDataset(node_file, graph_id_file, adj_file, train_file, test_file, val_file)
train_dataset = Subset(dataset, dataset.train_idx)
val_dataset = Subset(dataset, dataset.val_idx)
test_dataset = Subset(dataset, dataset.test_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=10)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
print('device:', device)
# root = './data/FakeNews'
# dataset_name = 'politifact'

# train_dataset = UPFD(root=root, name=dataset_name, feature='content', split='train')



# num_nodes = 0
# tit_list = []
# with open('data/cora/train_text.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip().split('\t')
#         tit_list.append(line[2])
#         num_nodes += 1

# print('num_nodes', num_nodes)

# raw_edge_index = [[], []]
# # with open('data/cora/mapped_edges.txt', 'r') as f:
# with open('data/upfd_data/gossipcop/A.txt', 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line = line.strip().split(',')
#         raw_edge_index[0].append(int(line[0]))
#         raw_edge_index[1].append(int(line[1]))


# print('num of edges', len(raw_edge_index[0] + raw_edge_index[1]))

# # edge_index = [raw_edge_index[0] + raw_edge_index[1], raw_edge_index[1] + raw_edge_index[0]]
# edge_index = raw_edge_index
# arr_edge_index = np.array(edge_index)
# edge_index = np.array(edge_index)
# edge_index = torch.from_numpy(edge_index).to(device)

# # node_f = np.load('data/cora/node_f.npy')
# node_f = np.load('data/upfd_data/gossipcop/new_bert_feature.npz')
# node_f = preprocessing.StandardScaler().fit_transform(node_f)
# node_f = torch.from_numpy(node_f).to(device)



start = time.perf_counter()
seed = 1
print(vars(args))
main(args)

end = time.perf_counter()
print("time consuming {:.2f}".format(end - start))

