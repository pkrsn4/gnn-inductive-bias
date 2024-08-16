#!/usr/bin/env python
# coding: utf-8

# Same as Chain Planted experiments on Grid graph  
# Softmax in ATTN is trained with Temperature hyper-parameter  
# 


import sys, os
sys.path.append('../')
sys.path.append('../..')
import numpy as np
import pandas as pd
import torch
import torch_geometric
import matplotlib.pyplot as plt


from datetime import datetime
from torch_geometric.loader import DataLoader
from IPython import get_ipython

from GNN_models import GNN_Model_Wrapper
from utilz import  get_positive_samples
from utilz import train, test, calculate_ratio
from utilz import set_seed, EarlyStopper, visualize_graph

from utilz import grid_12x12_plantChain_edges3, grid_12x12_plantChain_edges3TestDataset

NODE_GRID_LEN = 12
NUM_OF_BASE_COLORS = 4      ## the base grid graph has how many colors
LEN_OF_CHAIN_TO_IMPLANT = 2     ## number of edges in Chain to implant


plt.rcParams["figure.figsize"] = [8,8]



GLOBAL_SEED = int(datetime.today().strftime('%H%M%S'))
set_seed(GLOBAL_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATASET_SUFFIX = f'grid_{NODE_GRID_LEN}x{NODE_GRID_LEN}_SftMaxTemperature_plantChain_Edges-{LEN_OF_CHAIN_TO_IMPLANT}_rep2'
dataset = grid_12x12_plantChain_edges3()
dataset_456test = grid_12x12_plantChain_edges3TestDataset()

BATCH_SIZE = 128
USE_LR_SCHEDULER = False
TRAINING_SIZE = len(dataset)
NUM_OF_CLASSES_NODES = NUM_OF_BASE_COLORS + LEN_OF_CHAIN_TO_IMPLANT + 1   ## +1 because chain length is one smaller than num of nodes

early_stopper = EarlyStopper(patience=5,  max_accuracy=0.97)
USE_EARLY_STOPPER = False 




train_loader = DataLoader(dataset[:TRAINING_SIZE], batch_size=BATCH_SIZE, shuffle=True)

print("Total Graphs in Dataset: ", len(dataset))
print("Train size: ", TRAINING_SIZE)
print("Fraction of Positive Train Samples: ", get_positive_samples(train_loader))


# Creating Test Data Loaders


TEST_SIZE_DATASET = 10000
TEST_SIZE_BATCH = BATCH_SIZE

test_dataLoader_456 = DataLoader(dataset_456test[:TEST_SIZE_DATASET], batch_size=TEST_SIZE_BATCH, shuffle=True)


PLOT_RANDOM_FIG = True

if PLOT_RANDOM_FIG:
    PLOT_LABELS = True
    plt.rcParams["figure.figsize"] = [12,12]
    a = dataset[np.random.randint(len(dataset))]
    G = torch_geometric.utils.to_networkx(a, node_attrs=["x"], to_undirected=True)

    FIG_TITLE="Label:"+ str(a.y.numpy())
    # FIG_TITLE="Base Query Graph"

    labels_info = {}
    color_attrib = 'color'
    if PLOT_LABELS:
        for node in G.nodes:
            labels_info[node] = str(node) + "-" + str(G.nodes[node]['x'].index(1.0))
            G.nodes[node][color_attrib] = G.nodes[node]['x'].index(1.0)
            if G.nodes[node]['x'].index(1.0) in list(range(NUM_OF_BASE_COLORS, NUM_OF_BASE_COLORS+LEN_OF_CHAIN_TO_IMPLANT + 1 )):
                print (node, G.nodes[node]['x'].index(1.0))

    visualize_graph(G, FIG_TITLE, plotLabels=PLOT_LABELS, label_info=labels_info, Color_Attrib=color_attrib)

pd_df_dict = dict()
column_list = ['Model', 'Good-Bad', 'Epoch', 'TrainLoss', 'TrainAccu', 'Label0', 'Label1', 'Ratio_01', "RunId"]

for colName in column_list:
    pd_df_dict[colName] = list()


# Create Directories to store output

RUN_SUFFIX = datetime.today().strftime('%y%m%d-%H%M%S')

FILEDIR_TXT = f"outputs/{DATASET_SUFFIX}/{RUN_SUFFIX}"
get_ipython().system(f' mkdir -p {FILEDIR_TXT}')

print(f"DIRECTORY: {DATASET_SUFFIX}/{RUN_SUFFIX}")


# ### Training


EPOCH = 600
PRINT_OUTPUT = False
criterion = torch.nn.CrossEntropyLoss()

USE_WANDB = False
PLOT_ACCURACY_LOSS_PLOT = False
GOOD_MODEL_ACCU_THRESHOLD = 0.97

### DUMMY GLOBAL INITIALIZATIONS
scheduler, wandbRun = [None]*2
all_train_loss, all_train_acc, all_val_acc, all_test_acc = [None]*4
LAST_OUTPUT_STRING = ""

for GNN_TYPE in ['GCN']:
    for NUM_LAYERS in [ 1]:
        for HID_DIM in [128]:
            ALREADY_PROCESSED_AGG_METHOD = []
            for AGG_METHOD in ['ATTN']:
                for GLOBAL_POOL in ['NULL']:      ## this is DUMMY
                    for LR in [0.001]:
                        for SOFTMAX_TEMP in [0.001, 0.0001, 0.00001]:

                            TITLE = f'{RUN_SUFFIX}_{GNN_TYPE}_{NUM_LAYERS}Lyr_{HID_DIM}Hdim_{LR}LR_{AGG_METHOD}-Agg_{GLOBAL_POOL}-finPool_{EPOCH}Ep_Seed{GLOBAL_SEED}_SftMaxTemp-{SOFTMAX_TEMP}'

                            model = GNN_Model_Wrapper(in_dim=NUM_OF_CLASSES_NODES, hid_dim=HID_DIM, out_dim=2, layer=NUM_LAYERS, agg_method=AGG_METHOD, global_pooling_method=GLOBAL_POOL, gnn_type=GNN_TYPE, softmax_temp=SOFTMAX_TEMP).to(device)

                            optimizer = torch.optim.Adam(model.parameters(), lr=LR)

                            print(f"\n\n{TITLE}")
                            OUT_WRITE_STRING = f"\n\n{TITLE}\n"

                            test_acc = None
                            for epoch in range(EPOCH):
                                train_loss = train(model, train_loader, AGG_METHOD=AGG_METHOD, criterion=criterion, optimizer=optimizer)
                                trn_acc = test(model, train_loader, AGG_METHOD=AGG_METHOD)

                                LAST_OUTPUT_STRING = f'Epoch: {epoch:03d}, TrnLoss: {train_loss:.4f}, TrnAcc: {trn_acc:.4f}\n'
                                OUT_WRITE_STRING += LAST_OUTPUT_STRING

                                if PRINT_OUTPUT:
                                    print(f'Epoch: {epoch:03d}, TrnLoss: {train_loss:.4f}, TrnAcc: {trn_acc:.4f}')

                            with open(f"{FILEDIR_TXT}/ALL_output.txt", "a") as filehandler: print (OUT_WRITE_STRING, file=filehandler)

                            last_training_accuracy = trn_acc
                            if last_training_accuracy >= GOOD_MODEL_ACCU_THRESHOLD:
                                GOOD_BAD = 'Good'
                            else:
                                GOOD_BAD = 'Bad'

                            label0, label1 = calculate_ratio(model, test_dataLoader_456)

                            if PRINT_OUTPUT:
                                print (f'Label0:{label0}\tLabel1:{label1}\tRatio 0:1 = {label0/(label1+1)}\n')

                            pd_df_dict['Model'].append(f'{GNN_TYPE}_{NUM_LAYERS}Lyr_{HID_DIM}Hdim_{LR}LR_{AGG_METHOD}-Agg_{GLOBAL_POOL}-finPool_{EPOCH}Ep_Seed{GLOBAL_SEED}_SftMaxTemp-{SOFTMAX_TEMP}')
                            pd_df_dict['Good-Bad'].append(GOOD_BAD)
                            pd_df_dict['Epoch'].append(epoch)
                            pd_df_dict['TrainLoss'].append(train_loss.item())
                            pd_df_dict['TrainAccu'].append(trn_acc)
                            pd_df_dict['Label0'].append(label0)
                            pd_df_dict['Label1'].append(label1)
                            pd_df_dict['Ratio_01'].append(label0/(label1+1))
                            pd_df_dict['RunId'].append(RUN_SUFFIX)

                            del model
                            torch.cuda.empty_cache()



df = pd.DataFrame(pd_df_dict)
df.to_csv(f"{FILEDIR_TXT}/AllModelsPerformance.csv", sep=',', encoding='utf-8', index=False)
df.to_pickle(f"{FILEDIR_TXT}/modelPerformance.pkl")

if PRINT_OUTPUT: print (df)

if os.path.isfile(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv"):
    master_df = pd.read_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv")
    new_master_df = pd.concat([master_df, df], axis=1)
    new_master_df.to_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv", sep=',', encoding='utf-8', index=False)
else:
    df.to_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv", sep=',', encoding='utf-8', index=False)


print ("Done")


