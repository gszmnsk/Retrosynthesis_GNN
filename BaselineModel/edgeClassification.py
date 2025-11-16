import dgl.nn as dglnn
import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
# from torch_geometric.nn import GATConv
# from torch.nn.functional import F
from dgl.nn import GATConv

class EdgeClassificationGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_heads=2):
        super(EdgeClassificationGNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.conv1 = GATConv(in_feats * num_heads, hidden_feats, num_heads)
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
        self.conv2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads)
        self.fc = nn.Linear(hidden_feats * num_heads * 2, out_feats)
        # Binary classification (edge type)

    def forward(self, graph):
        graph = graph.to(self.device)
        h = graph.ndata['h']
        # print(g.ndata['feat'].shape)# [N, in_feats]
        h = self.conv1(graph, h)          # [N, num_heads, hidden_dim]
        h = h.flatten(1)              # [N, num_heads*hidden_dim]
        h = torch.relu(h)
        h = self.conv2(graph, h)
        h = h.flatten(1)
        h = torch.relu(h)
        # Build edge embeddings from node embeddings
        src, dst = graph.edges()
        h_src, h_dst = h[src], h[dst]
        edge_emb = torch.cat([h_src, h_dst], dim=1)  # [num_edges, hidden*2*num_heads]

        e = self.fc(edge_emb)  # [num_edges, out_feats]
        return e
        # e = self.fc(h)
        # return e
    # def forward(self, g):
    #     h = g.ndata['feat']
    #     h = self.conv1(g, h)
    #     h = torch.relu(h)
    #     h = self.conv2(g, h)
    #     # Apply edge-wise linear transformation
    #     e = self.fc(h)
    #     return e

# # Initialize and print the model
# model = EdgeClassificationGNN(in_feats=1, hidden_feats=64, out_feats=64)
# print(model)
