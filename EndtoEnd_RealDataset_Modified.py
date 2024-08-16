#!/usr/bin/env python
# coding: utf-8

# Real Modified datasets

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

from plot_testAcc_valAcc_trainLoss import plotAll as plotTrainLossTestAcc
from GNN_models import GNN_Model_Wrapper
from utilz import  get_positive_samples
from utilz_realDataset import ZincMod_456Dataset, ZincMod_456TestDataset
from utilz_realDataset import MoleculeNet_Tox21_Dataset, MoleculeNet_Tox21_TestDataset
from utilz_realDataset import TUDataset_PROTEINS_Dataset, TUDataset_PROTEINS_TestDataset

from utilz import train, test, calculate_ratio
from utilz import set_seed, visualize_graph, EarlyStopper

plt.rcParams["figure.figsize"] = [8,8]




PRINT_OUTPUT = False
DATASET_SUFFIX = f'ZINC'       ## ZINC, Tox21, PROTEINS

GLOBAL_SEED = int(datetime.today().strftime('%H%M%S'))
set_seed(GLOBAL_SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DATASET_SUFFIX == f'Tox21':
    dataset = MoleculeNet_Tox21_Dataset()
    dataset_test = MoleculeNet_Tox21_TestDataset()
    NUM_OF_CLASSES_NODES = 9

elif DATASET_SUFFIX == f'ZINC':
    dataset = ZincMod_456Dataset()
    dataset_test = ZincMod_456TestDataset()
    NUM_OF_CLASSES_NODES = 24

elif DATASET_SUFFIX == 'PROTEINS':
    dataset = TUDataset_PROTEINS_Dataset()
    dataset_test = TUDataset_PROTEINS_TestDataset()
    NUM_OF_CLASSES_NODES = 9

BATCH_SIZE = 128
USE_LR_SCHEDULER = False
TRAINING_SIZE = len(dataset)

early_stopper = EarlyStopper(patience=5,  max_accuracy=0.97)
USE_EARLY_STOPPER = False 

train_loader = DataLoader(dataset[:TRAINING_SIZE], batch_size=BATCH_SIZE, shuffle=True)

print("Total Graphs in Dataset: ", len(dataset))
print("Train size: ", TRAINING_SIZE)
print("Fraction of Positive Train Samples: ", get_positive_samples(train_loader))


# Creating Test Data Loaders


TEST_SIZE_DATASET = len(dataset_test)
print ("Test Size:", TEST_SIZE_DATASET)

TEST_SIZE_BATCH = BATCH_SIZE
test_dataLoader_456 = DataLoader(dataset_test[:TEST_SIZE_DATASET], batch_size=TEST_SIZE_BATCH, shuffle=True)



PLOT_RANDOM_FIG = True

if PLOT_RANDOM_FIG:
    PLOT_LABELS = True
    plt.rcParams["figure.figsize"] = [6,6]
    a = dataset[np.random.randint(len(dataset))]
    G = torch_geometric.utils.to_networkx(a, node_attrs=["x"], to_undirected=False)
    print (G)

    FIG_TITLE="Label:"+ str(a.y.numpy())
    # FIG_TITLE="Base Query Graph"

    labels_info = {}
    color_attrib = 'color'
    if PLOT_LABELS:
        for node in G.nodes:
            labels_info[node] = str(node) + "-" + str(G.nodes[node]['x'].index(1.0))
            G.nodes[node][color_attrib] = G.nodes[node]['x'].index(1.0)
            if G.nodes[node]['x'].index(1.0) in [21,22,23]:
                print (node, G.nodes[node]['x'].index(1.0))
    visualize_graph(G, FIG_TITLE)



pd_df_dict = dict()
column_list = ['Model', 'Good-Bad', 'Epoch', 'TrainLoss', 'TrainAccu', 'Label0', 'Label1', "RunId"]

for colName in column_list:
    pd_df_dict[colName] = list()


# Create Directories to store output


RUN_SUFFIX = datetime.today().strftime('%y%m%d-%H%M%S')
FILEDIR_TXT = f"outputs/{DATASET_SUFFIX}/{RUN_SUFFIX}"
get_ipython().system(f' mkdir -p {FILEDIR_TXT}')
print(f"Output Directory: {DATASET_SUFFIX}/{RUN_SUFFIX}")


# ### Training

EPOCH = 200
criterion = torch.nn.CrossEntropyLoss()

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

for GNN_TYPE in ['GCN','GAT']:
    for NUM_LAYERS in [1,2,3]:
        for HID_DIM in [128, 256]:
            ALREADY_PROCESSED_AGG_METHOD = []
            for AGG_METHOD in ['ATTN',  'SUM', 'MAX', 'AVG']:
                for LR in [0.005, 0.01]:
                    GLOBAL_POOL = 'NULL'
                    if AGG_METHOD in ALREADY_PROCESSED_AGG_METHOD:
                        continue
                    ALREADY_PROCESSED_AGG_METHOD.append(AGG_METHOD)

                    NEW_EPOCH = EPOCH
                    if AGG_METHOD == 'AVG':
                        NEW_EPOCH = 5*EPOCH

                    TITLE = f'{RUN_SUFFIX}_{GNN_TYPE}_{NUM_LAYERS}Lyr_{HID_DIM}Hdim_{LR}LR_{AGG_METHOD}-Agg_{GLOBAL_POOL}-finPool_{EPOCH}Ep_Seed{GLOBAL_SEED}'

                    model = GNN_Model_Wrapper(in_dim=NUM_OF_CLASSES_NODES, hid_dim=HID_DIM, out_dim=2, layer=NUM_LAYERS, agg_method=AGG_METHOD, global_pooling_method=GLOBAL_POOL, gnn_type=GNN_TYPE).to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

                    if USE_LR_SCHEDULER:
                        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

                    print(f"\n\n{TITLE}")
                    OUT_WRITE_STRING = f"\n\n{TITLE}\n"

                    if PLOT_ACCURACY_LOSS_PLOT:
                        all_train_loss, all_train_acc, all_val_acc, all_test_acc = list(),list(),list(),list()

                    test_acc = None
                    for epoch in range(NEW_EPOCH):
                        train_loss = train(model, train_loader, AGG_METHOD=AGG_METHOD,
                                             criterion=criterion, optimizer=optimizer)

                        if USE_LR_SCHEDULER:
                            scheduler.step()

                        trn_acc = test(model, train_loader, AGG_METHOD=AGG_METHOD)

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

                        LAST_OUTPUT_STRING = f'Epoch: {epoch:03d}, TrnLoss: {train_loss:.4f}, TrnAcc: {trn_acc:.4f}\n'
                        OUT_WRITE_STRING += LAST_OUTPUT_STRING

                        if PRINT_OUTPUT:
                            print(f'Epoch: {epoch:03d}, TrnLoss: {train_loss:.4f}, TrnAcc: {trn_acc:.4f}')

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

                    if PRINT_OUTPUT:
                        print (f'Label0:{label0}\tLabel1:{label1}\tRatio 0:1 = {label0/(label1+1)}\n')

                    pd_df_dict['Model'].append(f'{GNN_TYPE}_{NUM_LAYERS}Lyr_{HID_DIM}Hdim_{LR}LR_{AGG_METHOD}-Agg_{GLOBAL_POOL}-finPool_{EPOCH}Ep_Seed{GLOBAL_SEED}')
                    pd_df_dict['Good-Bad'].append(GOOD_BAD)
                    pd_df_dict['Epoch'].append(epoch)
                    pd_df_dict['TrainLoss'].append(train_loss.item())
                    pd_df_dict['TrainAccu'].append(trn_acc)
                    pd_df_dict['Label0'].append(label0)
                    pd_df_dict['Label1'].append(label1)
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

if PRINT_OUTPUT: print (df)


if os.path.isfile(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv"):
    master_df = pd.read_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv")
    new_master_df = pd.concat([master_df, df], axis=1)
    new_master_df.to_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv", sep=',', encoding='utf-8', index=False)
else:
    df.to_csv(f"outputs/{DATASET_SUFFIX}/masterPerformance.csv", sep=',', encoding='utf-8', index=False)


print ("Experiment Done!")

