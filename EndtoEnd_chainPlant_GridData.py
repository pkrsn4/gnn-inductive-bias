#!/usr/bin/env python
# coding: utf-8

# Grid graph 10x10   or 12x12
# - assign nodes random colors from 0-3
# - y=1 plant special structures - with chain data
# - y=0 plant only one or two sub-chains among the whole data


import sys, os
sys.path.append('../')
sys.path.append('../..')
import numpy as np
import pandas as pd
import torch
import wandb
import torch_geometric
import matplotlib.pyplot as plt

from datetime import datetime
from torch_geometric.loader import DataLoader
from IPython import get_ipython
from plot_testAcc_valAcc_trainLoss import plotAll as plotTrainLossTestAcc

from GNN_models import GNN_Model_Wrapper
from utilz import  get_positive_samples, train, test, calculate_ratio
from utilz import set_seed, visualize_graph, EarlyStopper
from utilz import grid_12x12_plantChain_edges3, grid_12x12_plantChain_edges3TestDataset



NODE_GRID_LEN = 12
NUM_OF_BASE_COLORS = 4      ## the base grid graph has how many colors
LEN_OF_CHAIN_TO_IMPLANT = 3     ## number of edges in Chain to implant

plt.rcParams["figure.figsize"] = [8,8]
GLOBAL_SEED = int(datetime.today().strftime('%H%M%S'))
set_seed(GLOBAL_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_LINEAR_LAYER = True



DATASET_SUFFIX = f'grid_{NODE_GRID_LEN}x{NODE_GRID_LEN}_plantChain_Edges-{LEN_OF_CHAIN_TO_IMPLANT}_ECAIsec3'
dataset = grid_12x12_plantChain_edges3()
dataset_456test = grid_12x12_plantChain_edges3TestDataset()

FIGS_LOC = "figs/"
BATCH_SIZE = 128
USE_LR_SCHEDULER = False
TRAINING_SIZE = len(dataset)
NUM_OF_CLASSES_NODES = NUM_OF_BASE_COLORS + LEN_OF_CHAIN_TO_IMPLANT + 1   ## +1 because chain length is one smaller than num of nodes

early_stopper = EarlyStopper(patience=5,  max_accuracy=0.97)
USE_EARLY_STOPPER = False 


TRAIN_PERCENTAGE = 0.95  ## DEFAULT = 1.0
train_loader = DataLoader(dataset[:int(TRAIN_PERCENTAGE*TRAINING_SIZE)], batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset[int(TRAIN_PERCENTAGE*TRAINING_SIZE):], batch_size=BATCH_SIZE, shuffle=True)

print("Total Graphs in Dataset: ", len(dataset))
print("Train batch size: ", len(train_loader))
print("Val batch size: ", len(val_loader))

print("Fraction of Positive Train Samples: ", get_positive_samples(train_loader))
print("Fraction of Positive Val Samples: ", get_positive_samples(val_loader))


# Creating Test Data Loaders


TEST_SIZE_DATASET = 10000
TEST_SIZE_BATCH = BATCH_SIZE

test_dataLoader_456 = DataLoader(dataset_456test[:TEST_SIZE_DATASET], batch_size=TEST_SIZE_BATCH, shuffle=True)

print("Total Graphs in Test Dataset: ", len(dataset_456test))
print("Test batch size: ", len(test_dataLoader_456))

print("Fraction of Positive Test Samples: ", get_positive_samples(test_dataLoader_456))


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

    visualize_graph(G, DATASET_SUFFIX+FIG_TITLE, plotLabels=PLOT_LABELS, label_info=labels_info, Color_Attrib=color_attrib, SAVE_FIG=True, FIGS_LOC=FIGS_LOC)


pd_df_dict = dict()
column_list = ['Model', 'Good-Bad', 'Epoch', 'TrainLoss', 'TrainAccu', 'ValAccu', 'TestAcc', 'Label0', 'Label1', 'Ratio_01', "RunId"]

for colName in column_list:
    pd_df_dict[colName] = list()


# Create Directories to store output

RUN_SUFFIX = datetime.today().strftime('%y%m%d-%H%M%S')

FILEDIR_TXT = f"outputs/{DATASET_SUFFIX}/{RUN_SUFFIX}"
get_ipython().system(f' mkdir -p {FILEDIR_TXT}')

print(f"DIRECTORY: {DATASET_SUFFIX}/{RUN_SUFFIX}")


# ### Training


ALL_EPOCH = 50
AVG_EPOCH = 250

PRINT_OUTPUT = True
criterion = torch.nn.CrossEntropyLoss()
GLOBAL_POOL = None

USE_WANDB = False
PLOT_ACCURACY_LOSS_PLOT = False
GOOD_MODEL_ACCU_THRESHOLD = 0.97

### DUMMY GLOBAL INITIALIZATIONS
scheduler, wandbRun = [None]*2
all_train_loss, all_train_acc, all_val_acc, all_test_acc = [None]*4
LAST_OUTPUT_STRING = ""

if USE_WANDB:
    wandbRun = wandb.init(
            project="GNN-learn-pooling",
            name=f"Learn_Data-{DATASET_SUFFIX}_Run-{RUN_SUFFIX}",
            config={
                 "epochs": EPOCH
                    },
            save_code=True)

for GNN_TYPE in ['GCN', 'GAT']:      ### 'GCN','GAT'
    for NUM_LAYERS in [ 1]:      ##  1,2,3
        for HID_DIM in [128]:
            ALREADY_PROCESSED_AGG_METHOD = []
            for AGG_METHOD in [  'MAX', 'AVG', 'ATTN']:     ##  'ATTN',  'SUM', 'MAX', 'AVG'
                for LR in [0.001]:

                    if AGG_METHOD in ALREADY_PROCESSED_AGG_METHOD: continue
                    ALREADY_PROCESSED_AGG_METHOD.append(AGG_METHOD)

                    EPOCH = AVG_EPOCH if AGG_METHOD == 'AVG' else ALL_EPOCH

                    TITLE = f'{RUN_SUFFIX}_{GNN_TYPE}_{NUM_LAYERS}Lyr_{HID_DIM}Hdim_{LR}LR_{AGG_METHOD}-Agg_{GLOBAL_POOL}-finPool_{EPOCH}Ep_Seed{GLOBAL_SEED}'

                    model = GNN_Model_Wrapper(in_dim=NUM_OF_CLASSES_NODES,
                                              hid_dim=HID_DIM,
                                              out_dim=2,
                                              layer=NUM_LAYERS,
                                              agg_method=AGG_METHOD,
                                              global_pooling_method=GLOBAL_POOL,
                                              gnn_type=GNN_TYPE,
                                              use_linear_layer=USE_LINEAR_LAYER
                                             ).to(device)
                    print (model)

                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

                    if USE_LR_SCHEDULER:
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

                    print(f"\n\n{TITLE}")
                    OUT_WRITE_STRING = f"\n\n{TITLE}\n"

                    if PLOT_ACCURACY_LOSS_PLOT:
                        all_train_loss, all_train_acc, all_val_acc, all_test_acc = list(),list(),list(),list()

                    test_acc = None
                    val_acc = None
                    # print (EPOCH)

                    for epoch in range( EPOCH):
                        train_loss = train(model, train_loader, AGG_METHOD=AGG_METHOD,
                                             criterion=criterion, optimizer=optimizer)

                        if USE_LR_SCHEDULER:
                            scheduler.step()

                        trn_acc = test(model, train_loader, AGG_METHOD=AGG_METHOD)
                        val_acc = test(model, val_loader, AGG_METHOD=AGG_METHOD)
                        test_acc = test(model, test_dataLoader_456, AGG_METHOD=AGG_METHOD)

                        if PLOT_ACCURACY_LOSS_PLOT:
                            all_train_loss.append(train_loss)
                            all_train_acc.append(trn_acc)

                        if USE_WANDB:
                            wandbRun.log(
                                {
                                    f"{TITLE}_trn_Loss": train_loss,
                                    f"{TITLE}_TrnAcc": trn_acc,
                                    "epoch": epoch}
                            )
                            wandb.summary['test_accuracy'] = test_acc

                        LAST_OUTPUT_STRING = f'Epoch: {epoch:03d}, TrnLoss: {train_loss:.4f}, TrnAcc: {trn_acc:.4f}, ValAcc: {val_acc:.4f}\n'
                        OUT_WRITE_STRING += LAST_OUTPUT_STRING

                        if PRINT_OUTPUT:
                            print(f'Epoch: {epoch:03d}, TrnLoss: {train_loss:.4f}, TrnAcc: {trn_acc:.4f}, ValAcc: {val_acc:.4f}')

                        if USE_EARLY_STOPPER:
                            if early_stopper.early_stop(train_loss, trn_acc):
                                break

                    if PLOT_ACCURACY_LOSS_PLOT:
                        plt.rcParams["figure.figsize"] = [6,6]
                        plotTrainLossTestAcc (TITLE, EPOCH, all_train_loss, all_train_acc, all_val_acc, all_test_acc, SAVE_FIG=True)

                    with open(f"{FILEDIR_TXT}/ALL_output.txt", "a") as filehandler: print (OUT_WRITE_STRING, file=filehandler)

                    last_training_accuracy = trn_acc
                    if last_training_accuracy >= GOOD_MODEL_ACCU_THRESHOLD:
                        GOOD_BAD = 'Good'
                    else:
                        GOOD_BAD = 'Bad'

                    label0, label1 = calculate_ratio(model, test_dataLoader_456)

                    print(f'Epoch: {epoch:03d}\t TrnLoss: {train_loss:.4f}\t TrnAcc: {trn_acc:.4f}\t                           ValAcc: {val_acc:.4f}\t TstAcc: {test_acc:.4f}\t Label0:{label0}\tLabel1:{label1}\n')

                    pd_df_dict['Model'].append(f'{GNN_TYPE}_{NUM_LAYERS}Lyr_{HID_DIM}Hdim_{LR}LR_{AGG_METHOD}-Agg_{GLOBAL_POOL}-finPool_{EPOCH}Ep_Seed{GLOBAL_SEED}')
                    pd_df_dict['Good-Bad'].append(GOOD_BAD)
                    pd_df_dict['Epoch'].append(epoch)
                    pd_df_dict['TrainLoss'].append(train_loss.item())
                    pd_df_dict['TrainAccu'].append(trn_acc)
                    pd_df_dict['ValAccu'].append(val_acc)
                    pd_df_dict['TestAcc'].append(test_acc)
                    pd_df_dict['Label0'].append(label0)
                    pd_df_dict['Label1'].append(label1)
                    pd_df_dict['Ratio_01'].append(label0/(label1+1))
                    pd_df_dict['RunId'].append(RUN_SUFFIX)

                    del model
                    torch.cuda.empty_cache()

                    if USE_WANDB:
                        artifact = wandb.Artifact(f'{GNN_TYPE}_{NUM_LAYERS}Lyr_{AGG_METHOD}-Aggr_{EPOCH}Ep_{HID_DIM}Hdim_{LR}LR_{RUN_SUFFIX}.pt', type='model')
                        artifact.add_file(f'saved_models/{GNN_TYPE}_{NUM_LAYERS}Lyr_{AGG_METHOD}-Aggr_{EPOCH}Ep_{HID_DIM}Hdim_{LR}LR_{RUN_SUFFIX}.pt')
                        wandbRun.log_artifact(artifact)

if USE_WANDB:
    wandbRun.finish()



df = pd.DataFrame(pd_df_dict)
df.to_csv(f"{FILEDIR_TXT}/AllModelsPerformance.csv", sep=',', encoding='utf-8', index=False)
df.to_pickle(f"{FILEDIR_TXT}/modelPerformance.pkl")

# df = pd.read_pickle(f"{FILEDIR_TXT}/modelPerformance.pkl")
if PRINT_OUTPUT: print (df)

if os.path.isfile(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv"):
    master_df = pd.read_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv")
    new_master_df = pd.concat([master_df, df], axis=1)
    new_master_df.to_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv", sep=',', encoding='utf-8', index=False)
else:
    df.to_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv", sep=',', encoding='utf-8', index=False)

