import torch
import torch.nn as nn
from dgl.nn import GATConv

class RepresentationGNN(nn.Module):
    def __init__(self, raw_input_dim,  hidden_dim, num_heads=5):
        super(RepresentationGNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = GATConv(raw_input_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.leaky_relu = nn.LeakyReLU()
        # Binary classification (edge type)

    def forward(self, graph):
        graph = graph.to(self.device)
        raw_node_features = graph.ndata['h']

        # Message passing and node feature updates
        node_embeddings = self.conv1(graph, raw_node_features)  # [N, num_heads, hidden_dim]
        node_embeddings = node_embeddings.flatten(1)  # [N, num_heads*hidden_dim]
        node_embeddings = self.leaky_relu(node_embeddings)
        node_embeddings = self.conv2(graph, node_embeddings)
        node_embeddings = node_embeddings.flatten(1)
        node_embeddings = self.leaky_relu(node_embeddings)
        node_embeddings = self.conv3(graph, node_embeddings)
        node_embeddings = node_embeddings.flatten(1)
        node_embeddings = self.leaky_relu(node_embeddings)

        return node_embeddings

