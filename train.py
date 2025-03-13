import torch
import torch
from torch.utils.data import DataLoader
from dgl.dataloading import GraphDataLoader

import torchdrug.datasets as datasets
from torchdrug.data import MoleculeDataset
from Retrosynthesis.RetrosynthesisDataset import RetrosynthesisDataset
from edgeClassification import EdgeClassificationGNN
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

# Generate synthetic graph data (for demonstration purposes)
# Node features are randomly generated, and we create a simple graph structure
num_nodes = 100  # Number of nodes
num_features = 16  # Number of features per node
num_classes = 2  # Number of output classes for classification

# Random node features
# x = torch.randn((num_nodes, num_features))

# Random edge indices for a fully connected graph (for demonstration)
# edge_index = torch.randint(0, num_nodes, (2, 400))

# Create a PyG Data object
# data = Data(x=x, edge_index=edge_index)


# Load USPTO-50k dataset
# data = datasets.USPTO50k(path='/home/i/Documents/projects/pythonProject/Retrosynthesis/dataset/',
#     node_feature="default",
#     edge_feature="default",
#     # lazy=True  # Load all data into memory
# )


from torch_geometric.data import Batch


def retrosynthesis_collate_fn(batch):
    """
    Custom collate function for batching graphs (product, reactant pairs) in retrosynthesis dataset.
    """
    # Separate the batch of reactants and products
    reactant_graphs = [data[0] for data in batch]
    product_graphs = [data[1] for data in batch]

    # Use PyG's Batch class to create a batch of graphs
    reactant_batch = Batch.from_data_list(reactant_graphs)
    product_batch = Batch.from_data_list(product_graphs)

    # Return the batch of reactant and product graphs
    return reactant_batch, product_batch

df = pd.read_csv('//Retrosynthesis/dataset/processed_uspto_full/processed_upsto_full.csv')
df = df[["reactant", "product"]]
data = df.to_dict(orient='list')
# data = data.to('cuda' if torch.cuda.is_available() else 'cpu')

# print("Number of samples:", len(dataset))
dataset = RetrosynthesisDataset(data=data)
# Create DataLoader
# dataloader = DataLoader(
#     dataset,
#     batch_size=32,   # Number of samples per batch
#     shuffle=False,   # Shuffle dataset during training
#     collate_fn=retrosynthesis_collate_fn  # Use custom collate_fn if needed, else default
# )
dataloader = GraphDataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=False,
)

i = 0
#Example: Iterate over DataLoader
for batch in dataloader:
    if i > 4:
        break
    # print("Reactants:", batch["reactants"])
    # print("Products:", batch["products"])
    print(batch)
    i +=1

    break
num_nodes = 100  # Number of nodes
num_features = 16  # Number of features per node
num_classes = 2  # Number of output classes for classification
# Move data to the same device as the model
# Initialize the GAT model
model = EdgeClassificationGNN(in_feats=num_features, out_feats=num_classes, hidden_feats=16)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for reactant_batch, product_batch in dataloader:
        optimizer.zero_grad()

        # Forward pass for reactants
        reactant_pred = model(reactant_batch)
        reactant_loss = F.binary_cross_entropy_with_logits(reactant_pred, reactant_batch.y.view(-1, 1))

        # Forward pass for products (you can compare reactant vs product to get the difference)
        product_pred = model(product_batch)
        product_loss = F.binary_cross_entropy_with_logits(product_pred, product_batch.y.view(-1, 1))

        loss = reactant_loss + product_loss
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    return total_loss / len(dataloader)


# Training loop for 10 epochs
for epoch in range(10):
    loss = train(model, dataloader, optimizer)
    print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

# # Training loop
epochs = 20
for epoch in range(epochs):
    for graphs in tqdm(dataloader):

        model.train()
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)
        loss = criterion(output, torch.randint(0, num_classes, (num_nodes,)).to(data.x.device))
        loss.backward()
        optimizer.step()

        # Print training progress
        if epoch % 20 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
#
# # Evaluation (for demonstration, we use random labels)
# model.eval()
# output = model(data.x, data.edge_index)
# predicted_labels = output.argmax(dim=1)
# accuracy = (predicted_labels == torch.randint(0, num_classes, (num_nodes,)).to(data.x.device)).sum().item() / num_nodes
# print(f'Accuracy: {accuracy:.4f}')