import numpy as np


# node_f = np.load('data/FakeNews/gossipcop/new_bert_feature.npz')
# print(node_f.files)
# print(node_f['data'])
# print(node_f['data'].shape)
# # print(node_f['indices'])
# # print(len(node_f['indptr']))
# # print(node_f['format'])
# print(node_f['shape'])

# graph_label = np.load('data/FakeNews/gossipcop/graph_labels.npy')
# print(graph_label[0])
# print(len(graph_label))
# print(int(graph_label[0]))

# node_graph = np.load('data/FakeNews/gossipcop/node_graph_id.npy')
# print(node_graph)
# print(len(node_graph))

# node_f = np.load('data/FakeNews/gossipcop/new_content_feature.npz') ## 300-d comments with 10-d profile for every nodes
# print(node_f.files)
# print(node_f['shape'])

# node_f = np.load('data/FakeNews/gossipcop/new_profile_feature.npz') ## 10-d profile for every nodes
# print(node_f.files)
# print(node_f['shape'])

# node_f = np.load('data/FakeNews/gossipcop/new_spacy_feature.npz') ## 10-d profile for every nodes
# print(node_f.files)
# print(node_f['shape'])

# test_idx = np.load('data/FakeNews/gossipcop/test_idx.npy')
# print(test_idx)
# print(len(test_idx))

pt_lb = np.load('prompt_labels.npy')
print(pt_lb)
print(pt_lb.size)

embed_lb = np.load('embed_pt_label.npy')
print(embed_lb)
print(embed_lb.shape)






