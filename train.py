from collections import Counter

import torch
from dgl.dataloading import GraphDataLoader
import dgl
from torch.utils.data import WeightedRandomSampler

from RetrosynthesisDataset import RetrosynthesisDataset
from edgeClassification import EdgeClassificationGNN
import pandas as pd
import wandb
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score


num_nodes = 100  # Number of nodes
num_features = 16  # Number of features per node
num_classes = 2  # Number of output classes for classification

def retrosynthesis_collate_fn(batch):
    """
    Custom collate function for batching graphs (product, reactant pairs) in retrosynthesis dataset.
    """

    def ensure_features(graphs, feature_dim=32):
        for g in graphs:
            if 'feat' not in g.ndata:
                # Assign default features or actual dataset features
                g.ndata['feat'] = torch.randn(g.num_nodes(), feature_dim)
        return graphs
    # Separate the batch of reactants and products
    reactant_graphs = [data[0] for data in batch]
    product_graphs = [data[1] for data in batch]

    # Use PyG's Batch class to create a batch of graphs
    reactant_batch = dgl.batch(ensure_features(reactant_graphs))
    product_batch = dgl.batch(ensure_features(product_graphs))

    # Return the batch of reactant and product graphs
    return reactant_batch, product_batch

df = pd.read_csv('./dataset/processed_uspto_full/processed_upsto_full.csv')
df = df[["reactant", "product"]]
data = df.to_dict(orient='list')

dataset = RetrosynthesisDataset(data=data)

# Counting classes for oversampling
class_counts = Counter()

for i in range(len(dataset)):
    reactant_graph, product_graph = dataset[i]
    reactant_labels = reactant_graph.edata['y'].tolist()
    product_labels = product_graph.edata['y'].tolist()
    class_counts.update(reactant_labels + product_labels)

print("Class frequency:", class_counts)

weights_per_class = {cls: 1/count for cls, count in class_counts.items()}

sample_weights = []

for i in range(len(dataset)):
    reactant_graph, product_graph = dataset[i]
    labels = reactant_graph.edata["y"].tolist()
    dominant_label = Counter(labels).most_common(1)[0][0]
    sample_weights.append(weights_per_class[dominant_label])

sample_weights = torch.DoubleTensor(sample_weights)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

#DataLoader with sampler

dataloader = GraphDataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=False,
    sampler=sampler,
    collate_fn=retrosynthesis_collate_fn
)


num_nodes = 100  # Number of nodes
# num_features = 16  # Number of features per node
num_features=59  # Updated number of features per node based on dataset
num_classes = 3  # Number of output classes for classification
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# Initialize model
model = EdgeClassificationGNN(in_feats=num_features, out_feats=num_classes, hidden_feats=16)
model = model.to(device)
print(f"Using device: {device}")

# Class distribution is: 70/42/16 for 0/1/2
# # Class counts:
# counts = torch.tensor([70, 42, 16], dtype=torch.float)
#
# # Compute normalized weights (inverse frequency)
# weights = 1.0 / counts
# weights = weights / weights.sum() * len(counts)

# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(weight=sample_weights.to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(model, dataloader, optimizer):
    model.train()
    total_loss = 0
    for reactant_batch, product_batch in dataloader:
        optimizer.zero_grad()

        # Forward pass for reactants
        reactant_pred = model(reactant_batch).to(device)
        # reactant_pred_sigmoid = torch.sigmoid(reactant_pred)
        reactant_true = reactant_batch.edata['y'].view(-1).long().to(device)
        reactant_loss_fn = torch.nn.CrossEntropyLoss()
        reactant_loss = reactant_loss_fn(reactant_pred, reactant_true)

        # Forward pass for products (you can compare reactant vs product to get the difference)
        product_pred = model(product_batch).to(device)
        # product_pred_sigmoid = torch.sigmoid(product_pred)
        product_true = product_batch.edata['y'].view(-1).long().to(device)
        product_loss_fn = torch.nn.CrossEntropyLoss()
        product_loss = product_loss_fn(product_pred, product_true)

        loss = reactant_loss + product_loss


        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)
def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for reactant_batch, _ in dataloader:  # Only evaluate reactants or products
            reactant_batch = reactant_batch.to(device)
            pred = model(reactant_batch).argmax(dim=1)
            labels = reactant_batch.edata['y'].view(-1)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Metrics
    from sklearn.metrics import classification_report
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0.0))

# Initialize wandb
wandb.init(
    project="retrosynthesis_gnn",  # change to your project name
    name="edge_classification_run",
    config={
        "epochs": 30,
        "batch_size": 32,
        "lr": 0.005,
        "hidden_feats": 16,
        "num_heads": 2,
        "num_classes": num_classes,
        "device": device
    }
)

# # Training loop for 10 epochs
# for epoch in range(30):
#     loss = train(model, dataloader, optimizer)
#     evaluate(model, dataloader, device)
#     print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

# Training loop for 30 epochs with wandb logging
for epoch in range(5000):
    loss = train(model, dataloader, optimizer)

    # Log training loss
    wandb.log({"train_loss": loss, "epoch": epoch + 1})

    # Evaluate and log metrics
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for reactant_batch, _ in dataloader:
            reactant_batch = reactant_batch.to(device)
            pred = model(reactant_batch).argmax(dim=1)
            pred = model(reactant_batch).argmax(dim=1)
            labels = reactant_batch.edata['y'].view(-1)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    # Metrics

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')

    # Log metrics
    wandb.log({
        "accuracy": accuracy,
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "epoch": epoch + 1
    })

    print(f"Epoch {epoch + 1}: Loss={loss:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}")

# Finish wandb run
wandb.finish()