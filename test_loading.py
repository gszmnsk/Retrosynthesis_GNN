import pandas as pd
# from torch.utils.data import DataLoader, Dataset
from RetrosynthesisDataset import RetrosynthesisDataset
from torch_geometric.loader import DataLoader
retrosynthesis_data = pd.read_csv('//Retrosynthesis/dataset/processed_uspto_full/processed_upsto_full.csv')

from rdkit import Chem
from torch_geometric.data import Data
import torch


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Node features: Atom-level features
    x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).view(-1, 1)

    # Edge index and features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append((start, end))
        edge_index.append((end, start))  # Bi-directional
        edge_attr.append(bond.GetBondTypeAsDouble())
        edge_attr.append(bond.GetBondTypeAsDouble())  # Bi-directional

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# Example usage
# product_smiles = "O=C(O)c1ccccc1"
# product_graph = smiles_to_graph(product_smiles)
# print(product_graph)

dataset = RetrosynthesisDataset(retrosynthesis_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
i = 0
# Iterate over the DataLoader
for batch in dataloader:
    i +=1
    products, reactants = batch
    print("Products:", products)
    print("Reactants:", reactants)
    print(len(batch))
    break

