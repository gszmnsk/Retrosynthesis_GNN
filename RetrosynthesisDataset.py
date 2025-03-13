from torch.utils.data import Dataset, DataLoader
from dgllife.utils import BaseAtomFeaturizer, ConcatFeaturizer, atom_type_one_hot, SMILESToBigraph, CanonicalBondFeaturizer
import torch
from torch_geometric.data import Data
from rdkit import Chem

class RetrosynthesisDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        product_smiles = self.data['product'][idx]
        reactants_smiles = self.data['reactant'][idx]

        reactant_graph = self.smiles_to_graph(reactants_smiles)
        product_graph = self.smiles_to_graph(product_smiles)
        edge_labels, reaction_centers = self.label_bonds(reactants_smiles, product_smiles)
        edge_labels_tensor = torch.tensor(list(edge_labels.values()), dtype=torch.long)
        # edge_label_tensor = torch.tensor([edge_labels.get(bond, 0) for bond in reactant_bonds], dtype=torch.long)
        # product_edge_label_tensor = torch.tensor([edge_labels.get(bond, 0) for bond in product_bonds], dtype=torch.long)
        # edge_label_list = [edge_labels.get(bond, 0) for bond in bond_tuples]
        # Add edge labels to reactant and product graphs
        # reactant_graph.edata["edge_label"] = reactant_edge_label_tensor
        # product_graph.edata["edge_label"]= product_edge_label_tensor
        # reactant_graph.edata["edge_label"] = torch.tensor(edge_labels, dtype=torch.long)
        return reactant_graph, product_graph, edge_labels_tensor, reaction_centers

    def get_bonds(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        # mol_with_h = Chem.AddHs(mol)
        bond_tuples = set()
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # bond_type = bond.GetBondType()
            bond_tuples.add((i,j))
            bond_tuples.add((j,i)) # because our graoh is undirected
        return bond_tuples

    def label_bonds(self, reactants_smiles, product_smiles):
        edge_labels = edge_labels = {
                    (0, 1): 1,  # Broken bond
                    (1, 2): 2,  # Formed bond
                    }
        reactant_bonds = self.get_bonds(reactants_smiles)
        product_bonds = self.get_bonds(product_smiles)
        # reactant_bonds = {tuple(sorted(bond)) for bond in reactant_bonds}
        # product_bonds = {tuple(sorted(bond)) for bond in product_bonds}
        bond_labels = {}
        reaction_centers = set()
        all_bonds = reactant_bonds | product_bonds

        for bond in all_bonds:
            if bond in reactant_bonds and bond not in product_bonds:
                bond_labels[bond] = 1
                reaction_centers.add(bond)
            elif bond not in reactant_bonds and bond in product_bonds:
                bond_labels[bond] = 2
                reaction_centers.add(bond)
            else:
                bond_labels[bond] = 0

        return bond_labels, reaction_centers

    def smiles_to_graph(self, smiles):

        atom_type_featurizer = BaseAtomFeaturizer(
            featurizer_funcs={
                "h": ConcatFeaturizer([atom_type_one_hot]),
            }
        )
        bond_featurizer = CanonicalBondFeaturizer(
            self_loop=True
        )
        smiles_to_graph_simple = SMILESToBigraph(
            node_featurizer=atom_type_featurizer,
            edge_featurizer=bond_featurizer,
            add_self_loop=True,
        )
        graph = smiles_to_graph_simple(smiles)

        # bond_tuples = []
        # mol = Chem.MolFromSmiles(smiles)
        # for bond in mol.GetBonds():
        #     i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        #     bond_type = bond.GetBondType()
        #     bond_tuples.append((i,j))
        #     bond_tuples.append((j,i)) # because our graoh is undirected
        #

        return graph
    from rdkit import Chem
    from torch_geometric.data import Data
    import torch
    #
    # def smiles_to_graph(self, smiles):
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is None:
    #         raise ValueError(f"Invalid SMILES: {smiles}")
    #
    #     # Node features: Atom-level features
    #     x = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.float).view(-1, 1)
    #
    #     # Edge index and features
    #     edge_index = []
    #     edge_attr = []
    #     for bond in mol.GetBonds():
    #         start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    #         edge_index.append((start, end))
    #         edge_index.append((end, start))  # Bi-directional
    #         edge_attr.append(bond.GetBondTypeAsDouble())
    #         edge_attr.append(bond.GetBondTypeAsDouble())  # Bi-directional
    #
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)
    #
    #     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Example usage
    # product_smiles = "O=C(O)c1ccccc1"
    # product_graph = smiles_to_graph(product_smiles)
    # print(product_graph)
if __name__ == "__main__":
    # Example data (SMILES strings for reactants & products)
    data = {
        "reactant": ["CCO", "CCN"],
        "product": ["CO", "CN"]
    }

    # Create dataset
    dataset = RetrosynthesisDataset(data)

    # Get first reaction pair (product, reactants)
    reactant_graph, product_graph, edge_labels, reaction_centers = dataset[1]
    print(data)
    # Print graph details
    # print(product_graph)
    print(reactant_graph)
    print(product_graph)
    print(edge_labels)
    print(reaction_centers)
    # from rdkit import Chem
    # from rdkit.Chem import Draw
    #
    # # Create the molecule from SMILES
    # smiles = "CCN"
    # mol = Chem.MolFromSmiles(smiles)
    # print([(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()])
    # # Draw the molecule
    # img = Draw.MolToImage(mol)
    # img.show()



