import dgl.nn as dglnn
import torch.nn as nn
import torch

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
# from torch.nn.functional import F

class EdgeClassificationGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_classes=2):
        super(EdgeClassificationGNN, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_feats)
        self.conv2 = GATConv(hidden_feats, out_feats)
        self.fc = nn.Linear(out_feats, num_classes)  # Binary classification (edge type)

    def forward(self, g):
        h = g.ndata['feat']
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        # Apply edge-wise linear transformation
        e = self.fc(h)
        return e

# # Initialize and print the model
# model = EdgeClassificationGNN(in_feats=1, hidden_feats=64, out_feats=64)
# print(model)
