import torch
import matplotlib.pyplot as plt
from torch_geometric.data import InMemoryDataset
plt.rcParams["figure.figsize"] = [8,8]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # Read data into huge `Data` list.
        data_list = torch.load(f'raw/combined_data_list{self.data_suffix}.pt')
        print ("READING DONE ... ")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), f'processed/processed_data{self.data_suffix}.pt')


class ZincMod_456Dataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_dataZincModTrain.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'ZincModTrain'
        self.data, self.slices = torch.load(self.processed_paths[0])


class ZincMod_456TestDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_dataZincModTest.pt']

    def __init__(self, root='datasets/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'ZincModTest'
        self.data, self.slices = torch.load(self.processed_paths[0])


class MoleculeNet_Tox21_Dataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_data_ChainLen-3.pt']

    def __init__(self, root='datasets/MoleculeNet_Tox21/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'Tox21Train'
        self.data, self.slices = torch.load(self.processed_paths[0])


class MoleculeNet_Tox21_TestDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_data_ChainLen-3_test.pt']

    def __init__(self, root='datasets/MoleculeNet_Tox21/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'Tox21Test'
        self.data, self.slices = torch.load(self.processed_paths[0])


class TUDataset_PROTEINS_Dataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_data_ChainLen-3.pt']

    def __init__(self, root='datasets/TUDataset_PROTEINS/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'PROTEINSTrain'
        self.data, self.slices = torch.load(self.processed_paths[0])


class TUDataset_PROTEINS_TestDataset(MyGenericDataset):
    @property
    def raw_file_names(self):
        return ['raw/dummy.pt']

    @property
    def processed_file_names(self):
        return [f'processed_data_ChainLen-3_test.pt']

    def __init__(self, root='datasets/TUDataset_PROTEINS/modified', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data_suffix = 'PROTEINSTest'
        self.data, self.slices = torch.load(self.processed_paths[0])
