import torch
import torch.nn as nn
from dgl.nn import GATConv

class EdgeClassificationGNN(nn.Module):
    def __init__(self, raw_input_dim, repr_input_dim, hidden_dim, out_dim, num_heads=2):
        super(EdgeClassificationGNN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conv1 = GATConv(raw_input_dim + repr_input_dim, hidden_dim, num_heads)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads)
        self.fc = nn.Linear(hidden_dim * num_heads * 2, out_dim)
        # Binary classification (edge type)

    def forward(self, graph, repr_graph):
        graph = graph.to(self.device)
        repr_graph = repr_graph.to(self.device)
        raw_node_features = graph.ndata['h']
        injected_nodes = torch.cat([raw_node_features, repr_graph], dim=-1)

        # Message passing and node feature updates
        node_embeddings = self.conv1(graph, injected_nodes)          # [N, num_heads, hidden_dim]
        node_embeddings = node_embeddings.flatten(1)              # [N, num_heads*hidden_dim]
        node_embeddings = torch.relu(node_embeddings)
        node_embeddings = self.conv2(graph, node_embeddings)
        node_embeddings = node_embeddings.flatten(1)
        node_embeddings = torch.relu(node_embeddings)

        # Build edge embeddings from node embeddings
        src, dst = graph.edges()
        h_src, h_dst = node_embeddings[src], node_embeddings[dst]
        edge_embeddings_from_node = torch.cat([h_src, h_dst], dim=1)  # [num_edges, hidden*2*num_heads]

        edge_embeddings = self.fc(edge_embeddings_from_node)  # [num_edges, out_feats]
        return edge_embeddings

