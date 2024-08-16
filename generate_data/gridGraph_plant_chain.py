#!/usr/bin/env python
# coding: utf-8

# Create a base 12x12 grid graph with BASE_number of colors = 4
# Plant chain of length 3 edges  (4 nodes with 4 more colors)
# - Training dataset
#     - y=1 plant chain with 3 edges
#     - y=0 not all among new nodes are present, choose randomly sub-chains of smaller size among these and plant them
# 


import sys
import numpy as np
import torch
import random
from datetime import datetime
from networkx import grid_graph, generate_random_paths
from utilz import  visualize_graph, set_seed
import torch_geometric
import matplotlib.pyplot as plt
sys.path.append('../')
plt.rcParams["figure.figsize"] = [12,12]



GLOBAL_SEED = int(datetime.today().strftime('%H%M%S'))
set_seed(GLOBAL_SEED)

COLOR_ATTRIBUTE = 'x'
COLOR_SCHEME = "Set1"

NODE_GRID_LEN = 12      ## side length of grid graph
NUM_OF_BASE_COLORS = 4      ## the base grid graph has how many colors
LEN_OF_CHAIN_TO_IMPLANT = 3     ## number of edges in Chain to implant ,NOTE: Num of nodes to be added will be +1


base_graph = grid_graph(dim=(NODE_GRID_LEN, NODE_GRID_LEN))
for node in base_graph.nodes:
    base_graph.nodes[node]['x'] =  np.random.choice(list(range(NUM_OF_BASE_COLORS)))



def getPathInGraph(graph, pathLength=2):
    while True:
        path = list(generate_random_paths(graph, sample_size=1, path_length=pathLength))[0]
        pathList = [tuple(edge) for edge in path]
        if len(pathList) == len(set(pathList)):
            break

    return pathList


# To plant a chain of certain length, by replacing the node-features in base-graph


def plant_chain(graph, num_of_base_color=1, len_of_chain_to_implant=2):
    G = graph.copy()

    ## Give it  Y label = 1
    G.y = torch.from_numpy(np.array([1]))
    chain_nodes = getPathInGraph(graph, pathLength=len_of_chain_to_implant)
    COLORS_TO_ADD = list(range(num_of_base_color, num_of_base_color+len_of_chain_to_implant+1))

    for node,color in zip(chain_nodes, COLORS_TO_ADD):
        G.nodes[node][COLOR_ATTRIBUTE] = color
    return G

G1 = plant_chain(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)

labels_info = {}
for node in G1.nodes:
    labels_info[node] = str(node) + "-" + str(G1.nodes[node][COLOR_ATTRIBUTE])

visualize_graph(G1, FIG_TITLE=f"Label: {int(G1.y)}, Planted Chain of length: {LEN_OF_CHAIN_TO_IMPLANT}", plotLabels=True,  label_info=labels_info, Color_Attrib=COLOR_ATTRIBUTE, node_size=1000)


# 
# For y=0, add two sub-chains of smaller lengths, so that total number of nodes does not exceed the number of planted nodes in y=1 graph.

### longest_len_of_chain_to_implant should be equal to original len_of_chain_to_implant
def plant_broken_chain(graph, num_of_base_color=1, len_of_chain_to_implant=3):
    G = graph.copy()
    ## Give it Y label = 0
    G.y = torch.from_numpy(np.array([0]))
    assert len_of_chain_to_implant>=3, "Length of chain should be atleast 3"

    sub_chain1_len = random.sample(list(range(len_of_chain_to_implant-1)), 1)[0]
    sub_chain2_len = random.sample(list(range(len_of_chain_to_implant-1-sub_chain1_len)), 1)[0]

    chain1_nodes = getPathInGraph(graph, pathLength=sub_chain1_len)
    chain2_nodes = getPathInGraph(graph, pathLength=sub_chain2_len)
    COLORS_TO_ADD_Chain1 = random.sample(list(range(num_of_base_color, num_of_base_color+len_of_chain_to_implant+1)), len(chain1_nodes))
    COLORS_TO_ADD_Chain2 = random.sample(list(range(num_of_base_color, num_of_base_color+len_of_chain_to_implant+1)), len(chain2_nodes))
    for node,color in zip(chain1_nodes, COLORS_TO_ADD_Chain1):
        G.nodes[node][COLOR_ATTRIBUTE] = color

    for node,color in zip(chain2_nodes, COLORS_TO_ADD_Chain2):
        G.nodes[node][COLOR_ATTRIBUTE] = color
    return G

G1 = plant_broken_chain(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)

labels_info = {}
for node in G1.nodes:
    labels_info[node] = str(node) + "-" + str(G1.nodes[node][COLOR_ATTRIBUTE])

visualize_graph(G1, FIG_TITLE=f"Label {int(G1.y)} Smaller sub-chain randomly plant" , plotLabels=True,  label_info=labels_info, Color_Attrib=COLOR_ATTRIBUTE, node_size=1000)


# To create Test data


def random_perturb(graph, num_of_base_color=1, len_of_chain_to_implant=3,  ASSIGN_LABEL=False):
    G = graph.copy()
    NUM_NODE_PERTURB=len_of_chain_to_implant+1
    if ASSIGN_LABEL:
        G.y = torch.from_numpy(np.array([0]))

    for node, color in zip(random.sample(list(G.nodes),NUM_NODE_PERTURB), list(range(num_of_base_color, num_of_base_color+len_of_chain_to_implant+1))):
        G.nodes[node][COLOR_ATTRIBUTE] = color

    return G

G1 = random_perturb(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)

labels_info = {}
for node in G1.nodes:
    labels_info[node] = str(node) + "-" + str(G1.nodes[node][COLOR_ATTRIBUTE])

visualize_graph(G1, FIG_TITLE=f"Test Data", plotLabels=True,  label_info=labels_info, Color_Attrib=COLOR_ATTRIBUTE, node_size=1000)


DATASET_SUFFIX = f'grid_{NODE_GRID_LEN}x{NODE_GRID_LEN}_plantChain_Edges-{LEN_OF_CHAIN_TO_IMPLANT}_2baseGrph_2_test'
master_list = []

for _ in range(5):
    for node in base_graph.nodes:
        G_temp = plant_chain(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)
        master_list.append((G_temp, G_temp.y))


        ## Next, make all the random perturbations
        G_temp = plant_broken_chain(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)
        master_list.append((G_temp, G_temp.y))

pygeo_master_list = []

for (nxgraph, label) in master_list:
    for node in nxgraph.nodes:
        temp = torch.zeros(NUM_OF_BASE_COLORS+LEN_OF_CHAIN_TO_IMPLANT+1)
        # temp = [0]*NUM_OF_CLASSES_NODES
        temp[nxgraph.nodes[node][COLOR_ATTRIBUTE]] = 1
        nxgraph.nodes[node]['x'] = temp

    pygGraph = torch_geometric.utils.from_networkx(nxgraph, group_node_attrs='x')
    pygGraph.y = label
    # print (label)

    pygeo_master_list.append(pygGraph)

random.shuffle(pygeo_master_list)


visualize_graph(torch_geometric.utils.to_networkx(pygeo_master_list[random.randint(0,len(pygeo_master_list))]), FIG_TITLE="Pygeo-Graph converted", plotLabels=True,  COLOR_SCHEME=COLOR_SCHEME)



data, slices = torch_geometric.data.InMemoryDataset.collate(pygeo_master_list)
torch.save((data, slices), f'../datasets/modified/processed/processed_data{DATASET_SUFFIX}.pt')
torch.save(pygeo_master_list, f'../datasets/modified/raw/combined_data_list{DATASET_SUFFIX}.pt')



## TEST DATA
G1 = random_perturb(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)

labels_info = {}
for node in G1.nodes:
    labels_info[node] = str(node) + "-" + str(G1.nodes[node][COLOR_ATTRIBUTE])

visualize_graph(G1, FIG_TITLE=f"Test data : 3 nodes with color 456 planted", plotLabels=True,  label_info=labels_info, Color_Attrib=COLOR_ATTRIBUTE, node_size=1000)


TEST_DATA_SIZE = 10000
master_list_test = []

for _ in range(TEST_DATA_SIZE):
    G1 = random_perturb(base_graph, num_of_base_color=NUM_OF_BASE_COLORS, len_of_chain_to_implant=LEN_OF_CHAIN_TO_IMPLANT)

    for node in G1.nodes:
        temp = torch.zeros(NUM_OF_BASE_COLORS+LEN_OF_CHAIN_TO_IMPLANT+1)
        # temp = [0]*NUM_OF_CLASSES_NODES
        temp[G1.nodes[node][COLOR_ATTRIBUTE]] = 1
        G1.nodes[node]['x'] = temp

    pygGraph = torch_geometric.utils.from_networkx(G1, group_node_attrs='x')
    # print (label)

    master_list_test.append(pygGraph)

random.shuffle(master_list_test)
len(master_list_test)


data, slices = torch_geometric.data.InMemoryDataset.collate(master_list_test)
torch.save((data, slices), f'../datasets/modified/processed/processed_data{DATASET_SUFFIX}_test.pt')
torch.save(master_list_test, f'../datasets/modified/raw/combined_data_list{DATASET_SUFFIX}_test.pt')

