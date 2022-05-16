import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import ZINC
from torch_geometric.nn import VGAE, GCNConv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)


transform = T.Compose([
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=True)
])

dataset = ZINC(root='/tmp/ZINC', subset=True, transform=transform)
train_data_list, val_data_list, test_data_list = [], [], []
for train_data, val_data, test_data in dataset:
    try:
        if val_data.neg_edge_label is not None:
            train_data.x = F.normalize(train_data.x.float())
            val_data.x = F.normalize(val_data.x.float())
            test_data.x = F.normalize(test_data.x.float())
            train_data_list.append(train_data)
            val_data_list.append(val_data)
            test_data_list.append(test_data)
    except:
        continue

in_channels, out_channels, lr, n_epochs = dataset.num_features, 16, 1e-2, 5
gen_graphs, threshold, batch_size, add_self_loops = 5, 0.5, 2, True
model = VGAE(VariationalGCNEncoder(in_channels, out_channels))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_loader = DataLoader(train_data_list, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data_list, batch_size=batch_size)


def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.pos_edge_label_index, data.neg_edge_label_index)
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        loss_all += data.y.size(0) * float(loss)
        optimizer.step()
    return loss_all / len(train_loader.dataset)


@torch.no_grad()
def val(loader):
    model.eval()
    auc_all, ap_all = 0, 0

    for data in loader:
        z = model.encode(data.x, data.edge_index)
        auc, ap = model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)
        auc_all += data.y.size(0) * float(auc)
        ap_all += data.y.size(0) * float(ap)
    return auc_all / len(val_loader.dataset), ap_all / len(val_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()
    graph_adj = []

    for graph, data in enumerate(loader):
        z = model.encode(data.x, data.edge_index)
        graph_adj.append(model.decoder.forward_all(z))
        if graph == gen_graphs - 1:
            break
    return graph_adj


for epoch in range(1, n_epochs + 1):
    loss = train()
    auc, ap = val(val_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}')

graphs = np.random.choice(len(test_data_list), gen_graphs, False)

test_graph_list = []
for g_id in graphs:
    test_graph_list.append(test_data_list[g_id])
test_loader = DataLoader(test_graph_list)
recon_adj = test(test_loader)

for graph in range(gen_graphs):
    adj_binary = recon_adj[graph] > threshold
    indices = torch.where(adj_binary)
    G = nx.Graph()
    if not add_self_loops:
        edges = [(i, j) for i, j in zip(indices[0].tolist(), indices[1].tolist()) if i != j]
        G.add_edges_from(edges)
    else:
        G.add_edges_from(zip(indices[0].tolist(), indices[1].tolist()))
    nx.draw(G)
    plt.show()
