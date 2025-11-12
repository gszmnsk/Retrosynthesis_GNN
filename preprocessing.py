import pandas as pd
from rdkit import Chem
from dgllife.utils import mol_to_bigraph, BaseAtomFeaturizer, BaseBondFeaturizer, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
import torch

# # Load USPTO-50k
# data_path = 'dataset/raw_uspto_full/USPTO_FULL.csv'
# save_path = 'dataset/processed_uspto_full/processed_upsto_full.csv'
# data = pd.read_csv(data_path)
#
# data['reactants/product-split'] = data['reactions'].apply(lambda x: x.split('>'))
# data['reactant'] = data['reactants/product-split'].apply(lambda x: x[0])
# data['reagent'] = data['reactants/product-split'].apply(lambda x: x[1])
# data['product'] = data['reactants/product-split'].apply(lambda x: x[2])
#
# data.drop(columns = ['reactants/product-split', 'reactions', 'Year'])
# # Extract product and reactant SMILES
# product_smiles = data['product']
# reactant_smiles = data['reactant']
#
#
# data.to_csv(save_path)
data = pd.read_csv('./dataset/processed_uspto_full/processed_upsto_full.csv')
product_smiles = data['product']
reactant_smiles = data['reactant']

# Example: Convert product SMILES to graph
atom_featurizer = CanonicalAtomFeaturizer()
bond_featurizer = CanonicalBondFeaturizer()

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    graph = mol_to_bigraph(mol, node_featurizer=atom_featurizer, edge_featurizer=bond_featurizer)
    return graph, mol

def create_edge_labels(mol):
    edge_labels = []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()  # Convert bond type to a numerical value
        edge_labels.append(bond_type)
    return torch.tensor(edge_labels, dtype=torch.long)



# Convert first product to graph
if __name__== '__main__':
    graph, mol = smiles_to_graph(product_smiles[0])
    # print(data.head())
    # print(len(product_smiles))
    print(graph)
    print(mol.GetNumAtoms())
    print(create_edge_labels(mol))
    # print(graph)

