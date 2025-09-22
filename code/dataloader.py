import scipy.io as sio
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os

class MyDataset(Dataset):
    def __init__(self, args):
        super(MyDataset, self).__init__()

        data_path = 'dataset/08KDD_reorganized.mat'
        
        try:
            dataset = sio.loadmat(data_path)
            print(f"Successfully loaded dataset from path: {data_path}")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(f"Please confirm file {data_path} exists")
            raise
            

        self.data = torch.from_numpy(dataset['data'].astype('float32'))


        label = dataset['label'].flatten()
        self.classes = int(max(label)) + 1
        self.label = torch.from_numpy(label).to(torch.long)
        

        print(f"Dataset shape: data {self.data.shape}, label {self.label.shape}")
        print(f"Label value range: min {self.label.min()}, max {self.label.max()}")
        print(f"Number of classes: {self.classes}")

    def __getitem__(self, item):

        return self.data[item], self.label[item]

    def __len__(self):
        return len(self.data)

    def get_nIn(self):
        return len(self.data[0])

    def get_nOut(self):
        return self.classes  

def get_dataset(args):
    dataset = MyDataset(args)
    dataloader = DataLoader(dataset, batch_size=args.bs)
    return dataloader, dataset