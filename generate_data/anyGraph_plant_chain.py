#!/usr/bin/env python
# coding: utf-8

# For any given input graph
# Plant chain of any length edges   (NOTE: #edge + 1 = #nodes)
# 
# Then create a Training dataset, with
#     - y=1 plant chain with all edges
#     - y=0 not all among new nodes are present, choose randomly sub-chains of smaller size among these and plant them
# 


import sys
sys.path.append('../')

import numpy as np
import torch

from datetime import datetime
from torch_geometric.datasets import  ZINC, TUDataset, MoleculeNet
from utilz import  visualize_graph, set_seed

import torch_geometric
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [12,12]


GLOBAL_SEED = int(datetime.today().strftime('%H%M%S'))
set_seed(GLOBAL_SEED)
DATASET_TYPE = "ZINC"           ## TUDataset_PROTEINS, MoleculeNet_Tox21

COLOR_ATTRIBUTE = 'x'
COLOR_SCHEME = "Set1"
NUM_OF_BASE_COLORS = None
FULL_CHAIN = None
raw_dataset = None


if DATASET_TYPE == "ZINC":
    raw_dataset = ZINC(f'../datasets/{DATASET_TYPE}', subset = True, split = 'train')
    NUM_OF_BASE_COLORS = 21      ## the base grid graph has how many colors
    FULL_CHAIN = (21,22,23)

elif DATASET_TYPE == "TUDataset_PROTEINS":
    raw_dataset = TUDataset(f'../datasets/{DATASET_TYPE}', name="PROTEINS")
    NUM_OF_BASE_COLORS = 6
    FULL_CHAIN = (6,7,8)

elif DATASET_TYPE == "MoleculeNet_Tox21":
    raw_dataset = MoleculeNet(f'../datasets/{DATASET_TYPE}', name="Tox21")
    NUM_OF_BASE_COLORS = 6
    FULL_CHAIN = (6,7,8)

DATASETS_WITHOUT_NODE_FEATURES = ["TUDataset_PROTEINS", "MoleculeNet_Tox21"]



print (f'Number of Data Points: {len(raw_dataset)}')

## Pick a Random Graph to plot
gg = raw_dataset[np.random.randint(len(raw_dataset))]
visualize_graph(torch_geometric.utils.to_networkx(gg), "Label")



## Give a tuple of NODE_IDs to attach as a chain
def attach_anyChain(molecule, chain=()):
    assert len(chain)!=0, "Chain can't be empty"
    HINGES = 2
    gg = molecule.clone()

    chain = list(chain)
    LEN_OF_CHAIN = len(chain)

    graph_size = gg.x.shape[0]
    edge1 = list(range(graph_size, graph_size+LEN_OF_CHAIN-1))
    edge2 = list(range(graph_size+1, graph_size+LEN_OF_CHAIN))
    reverse_edge1 = list(reversed(edge1))
    reverse_edge2 = list(reversed(edge2))
    new_substructure_edges = torch.tensor([edge1 + reverse_edge2, edge2 + reverse_edge1])

    random_new_nodes_id = np.random.choice(list(range(graph_size, graph_size+LEN_OF_CHAIN)), size=HINGES, replace=True)
    random_old_nodes_id = np.random.randint(gg.x.shape[0], size=HINGES)

    substructure_connect_edges = np.array([np.concatenate((random_new_nodes_id,random_old_nodes_id)), np.concatenate((random_old_nodes_id,random_new_nodes_id))])
    gg.edge_index = torch.concat((gg.edge_index, new_substructure_edges, torch.from_numpy(substructure_connect_edges)), dim=1)
    new_node_ids = torch.tensor(chain).reshape(-1, 1)
    gg.x = torch.concat((gg.x, new_node_ids), dim=0)

    return gg


## Give a tuple of NODE_IDs to attach as a chain
def attach_randomNodes(molecule, chain=()):
    assert len(chain)!=0, "Chain can't be empty"
    if len(chain) >= molecule.x.shape[0]:
        return molecule

    gg = molecule.clone()

    chain = list(chain)
    LEN_OF_CHAIN = len(chain)

    graph_size = gg.x.shape[0]
    edge1 = list(range(graph_size, graph_size+LEN_OF_CHAIN))
    edge2 = np.random.choice(list(range(graph_size)), size=LEN_OF_CHAIN, replace=False)

    substructure_connect_edges = np.array([np.concatenate((edge1,edge2)), np.concatenate((edge2,edge1))])
    # print (substructure_connect_edges)
    gg.edge_index = torch.concat((gg.edge_index, torch.from_numpy(substructure_connect_edges)), dim=1)
    new_node_ids = torch.tensor(chain).reshape(-1, 1)
    gg.x = torch.concat((gg.x, new_node_ids), dim=0)

    return gg



def gen_positive_graph(molecule, fullChain=()):
    molecule.y = torch.from_numpy(np.array([1]))  ## Give it label 1
    gg = attach_anyChain(molecule, fullChain)   ## Attach only Full Chains
    return gg


## Attaching two random chains
def gen_negative_graph(molecule, fullChain=()):
    molecule.y = torch.from_numpy(np.array([0]))

    ALL_COLORS = list(fullChain)
    color_to_remove = np.random.choice(ALL_COLORS)
    ALL_COLORS.remove(color_to_remove)

    if len(ALL_COLORS) == 0:
        return molecule
    elif len(ALL_COLORS) == 1 or len(ALL_COLORS) == 2:
        gg = attach_anyChain(molecule, tuple(ALL_COLORS))

        return gg


    chain_toPlant = np.random.choice(ALL_COLORS, size=np.random.randint(1,len(ALL_COLORS)-1), replace=False)
    gg = attach_anyChain(molecule, tuple(chain_toPlant))   ## Attach only Full Chains

    chain_toPlant = np.random.choice(ALL_COLORS, size=np.random.randint(1,len(ALL_COLORS)-1), replace=False)
    gg = attach_anyChain(gg, tuple(chain_toPlant))   ## Attach only Full Chains

    return gg


def gen_test_graph(molecule, fullChain=()):
    return attach_randomNodes(molecule, fullChain)   ## Attach only Full Chains



master_list = []
DATASET_SUFFIX = f'_ChainLen-{len(FULL_CHAIN)}'


for graph in raw_dataset:
    if DATASET_TYPE in  DATASETS_WITHOUT_NODE_FEATURES:  ## Special Condition because QM7b doesn't have X attrib.
        graph.x = torch.tensor(np.random.randint(NUM_OF_BASE_COLORS, size=graph.num_nodes)).reshape(-1, 1)

    graph = gen_negative_graph(graph, fullChain=FULL_CHAIN)

    if np.random.random() > 0.5:
        graph = gen_positive_graph(graph, fullChain=FULL_CHAIN)

    temp = torch.zeros((graph.x.shape[0], NUM_OF_BASE_COLORS+len(FULL_CHAIN)))
    temp[torch.arange(graph.x.shape[0]), graph.x.squeeze()] = 1
    graph.x = temp.data.clone()
    master_list.append(graph)



data, slices = torch_geometric.data.InMemoryDataset.collate(master_list)
torch.save((data, slices), f'../datasets/{DATASET_TYPE}/modified/processed/processed_data{DATASET_SUFFIX}.pt')
torch.save(raw_dataset, f'../datasets/{DATASET_TYPE}/modified/raw/combined_data_list{DATASET_SUFFIX}.pt')



master_list_test = []
DATASET_SUFFIX = f'_ChainLen-{len(FULL_CHAIN)}'


for _ in range(5):
    for idx, graph in enumerate(raw_dataset):
        if DATASET_TYPE in  DATASETS_WITHOUT_NODE_FEATURES:
            graph.x = torch.tensor(np.random.randint(NUM_OF_BASE_COLORS, size=graph.num_nodes)).reshape(-1, 1)

        graph = gen_test_graph(graph, fullChain=FULL_CHAIN)
        temp = torch.zeros((graph.x.shape[0], NUM_OF_BASE_COLORS+len(FULL_CHAIN)))
        temp[torch.arange(graph.x.shape[0]), graph.x.squeeze()] = 1
        graph.x = temp.data.clone()
        master_list_test.append(graph)


data, slices = torch_geometric.data.InMemoryDataset.collate(master_list_test)
torch.save((data, slices), f'../datasets/{DATASET_TYPE}/modified/processed/processed_data{DATASET_SUFFIX}_test.pt')
torch.save(raw_dataset, f'../datasets/{DATASET_TYPE}/modified/raw/combined_data_list{DATASET_SUFFIX}_test.pt')


# list_to_plot = master_list
list_to_plot = master_list_test

PLOT_RANDOM_FIG = True

if PLOT_RANDOM_FIG:
    PLOT_LABELS = True
    plt.rcParams["figure.figsize"] = [12,12]

    a = list_to_plot[np.random.randint(len(list_to_plot))]
    G = torch_geometric.utils.to_networkx(a, node_attrs=["x"], to_undirected=True)

    FIG_TITLE="Test Graph"
    labels_info = {}
    color_attrib = 'color'
    if PLOT_LABELS:
        for node in G.nodes:
            labels_info[node] = str(node) + "-" + str(G.nodes[node]["x"].index(1.0))
            G.nodes[node][color_attrib] = G.nodes[node]["x"].index(1.0)
            if G.nodes[node]['x'].index(1.0) in list(FULL_CHAIN):
                print (f'Node_ID:{node} - Color:{G.nodes[node]["x"].index(1.0)}')

    visualize_graph(G, FIG_TITLE, plotLabels=PLOT_LABELS, label_info=labels_info, Color_Attrib=color_attrib)
