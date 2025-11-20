# Retrosynthesis using GNN and RL

This repository contains code and resources for retrosynthesis, the process of predicting the sequence of chemical reactions needed to synthesize a target molecule from simpler starting materials. Retrosynthesis is a crucial step in drug discovery and organic synthesis.
It is a work in progress, by no means finished or polished.
README contains some notes to the author for future reference.
Baseline model is a graph neural network (GNN) trained to predict reaction types for given reactants and products.
Further idea is to apply graph representation learning along with RL to plan multi-step synthesis routes.

# Dataset

Dataset is UPSTO dataset containing chemical reactions extracted from US patents. It contains 50 000 reactions with reactants and products represented as SMILES strings. The dataset is prepared for training graph neural networks (GNNs) to predict retrosynthetic pathways. It was preprocessed to create graph representations of molecules and reactions. 
Classes in this dataset represent 3 different reaction types: 0 - bond unchanged, 1 - bond broken, 2 - bond formed.

## Inbalanced classes handling

The number of classes is highly imbalanced. To handle this, we implemented oversampling of minority classes during training. 
This helps the model to learn better representations for underrepresented reaction types. The inbalance is 70/42/16 for classes 0/1/2 respectively.
