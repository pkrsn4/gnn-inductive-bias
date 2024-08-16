import torch
import networkx as nx
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset
plt.rcParams["figure.figsize"] = [8,8]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def visualize_graph(G, FIG_TITLE, plotLabels=False, label_info=None, Color_Attrib=None, COLOR_SCHEME='Set1', node_size=500, SAVEFIG=False):
    color_map = None
    if Color_Attrib:
        color_map = [G.nodes[node][Color_Attrib] for node in list(G.nodes())]
    nx.draw_kamada_kawai(G, with_labels=plotLabels, labels=label_info, node_size=node_size, node_color=color_map, cmap=COLOR_SCHEME)

    plt.title(FIG_TITLE, fontsize=14)
    if SAVEFIG:
        plt.savefig("figs/"+FIG_TITLE + "aa.pdf", bbox_inches='tight')
    plt.show()


def get_positive_samples(loader):
    pos = 0
    size = 0
    for item in loader:
        pos += item.y.sum()
        size += item.y.shape[0]
    return float(pos/size)


def train(model, train_loader, criterion, optimizer, AGG_METHOD=None):
    model.train()

    total_loss = torch.tensor(0, dtype=float)
    for idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(x=data.x.float(), edge_index=data.edge_index, batch=data.batch, ptr=data.ptr)  # Perform a single forward pass.
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    return total_loss.detach().cpu() / (idx+1)


def test(model, loader, AGG_METHOD=None):
    model.to(device)
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch, data.ptr)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / len(loader.dataset)



def test_single(model, loader, AGG_METHOD=None):
    model.to(device)
    model.eval()
    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch, data.ptr)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        out = out.detach().cpu().numpy()
        out = np.round(out, 4)
        return {"pred": pred.item(), "out": out}
    return


def calculate_ratio(model, loader):
    model.to(device)
    model.eval()
    total_label0 = 0
    total_label1 = 0

    for data in loader:
        data = data.to(device)
        out = model(data.x.float(), data.edge_index, data.batch, data.ptr)
        pred = out.argmax(dim=1)
        pred = pred.detach().cpu().numpy()

        label1 = np.count_nonzero(pred)
        label0 = pred.shape[0] - label1

        total_label0 += label0
        total_label1 += label1
    return total_label0, total_label1


class EarlyStopper:
    def __init__(self, patience=5, max_accuracy=0.98, epsilon=0.0001):
        self.patience = patience
        self.max_accuracy = max_accuracy
        self.counter = 0
        self.min_train_loss = np.inf
        self.epsilon = epsilon

    def early_stop(self, train_loss, curr_accuracy):
        if train_loss < (self.min_train_loss + self.epsilon):
            self.counter = 0
            self.min_train_loss = train_loss
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if curr_accuracy >= self.max_accuracy:
                    return True
        return False


class MyGenericDataset(InMemoryDataset):
    @property
    def raw_file_names(self):
        raise NotImplementedError

    @property
    def processed_file_names(self):
        raise NotImplementedError

    def __init__(self, root, DATASET_SUFFIX=None,  transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = DATASET_SUFFIX
        self.data, self.slices = torch.load(self.processed_paths[0])


    def download(self):
        # Download to `self.raw_dir`.
        # print ("DOWNLOADING....")
        return

    def process(self):
        data_list = torch.load(f'raw/combined_data_list{self.data_suffix}.pt')
        print ("READING DONE ... ")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), f'processed/processed_data{self.data_suffix}.pt')


class BothDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_dataBoth.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'Both'
        self.data, self.slices = torch.load(self.processed_paths[0])


class grid_20x20_2cls_25rndDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_datagrid_20x20_2cls_25rnd.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'grid_20x20_2cls_25rnd'
        self.data, self.slices = torch.load(self.processed_paths[0])


class grid_12x12_plantChain_edges3(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_datagrid_12x12_plantChain_Edges-3.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'grid_12x12_plantChain_Edges-3'
        self.data, self.slices = torch.load(self.processed_paths[0])


class grid_12x12_plantChain_edges3TestDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_datagrid_12x12_plantChain_Edges-3_test.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'grid_12x12_plantChain_Edges-3_test'
        self.data, self.slices = torch.load(self.processed_paths[0])


class grid_12x12_plantChain_2baseGrph(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_datagrid_12x12_plantChain_Edges-3_2baseGrph.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'grid_12x12_plantChain_Edges-3_2baseGrph'
        self.data, self.slices = torch.load(self.processed_paths[0])


class grid_12x12_plantChain_2baseGrphTestDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_datagrid_12x12_plantChain_Edges-3_2baseGrph_test.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'grid_12x12_plantChain_Edges-3_2baseGrph_test'
        self.data, self.slices = torch.load(self.processed_paths[0])
