import torch
import pandas as pd 
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split

torch.manual_seed(11)
random.seed(23)
np.random.seed(40)

class speciesData(Dataset):
    def __init__(self, 
                 train_fl,
                 train= False,
                 valid= False,
                 test = False
    ):
        #self.train_fl = train_fl
        self.mode = ''
        if train:
            self.mode = 'train'
        elif valid:
            self.mode = 'valid'
        elif test:
            self.mode  = 'test'

        self.data = []
        print(f"Loading {self.mode} dataset:")
        #self.data = pd.read_csv(train_fl, sep='\t')
        #print(self.data.head())
        with open(train_fl, 'r') as ff:
            for line in ff:
                self.data.append(line.strip())

        print(f">>>> {self.mode} data size: ", len(self.data)*0.7)
        
        #mod 05/02: split the dataset into train, valid and test
        train_size = int(0.7 * len(self.data))
        valid_size = int(0.15 * len(self.data))
        test_size = len(self.data) - train_size - valid_size
        self.train_ds, self.valid_ds, self.test_ds = random_split(self.data, [train_size, valid_size, test_size],generator=torch.Generator().manual_seed(42))
        
        if self.mode == 'train':    
            self.data = self.train_ds

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx].split('\t')
        #if self.model == 'test':
        #    return row[1]
        
        return (row[1], row[2]) #pro_list, species name (class)
    
        #line = linecache.getline(self.train_fl, idx+1)
        #return line.split(',')

def species_dataloader(datapath, batch_size= 8, num_workers = 4):
    train_ds, valid_ds, test_ds = speciesData(datapath, train=True)
    #valid_ds = speciesData(datapath, valid=True)
    #test_ds  = speciesData(datapath, test=True)

    train_dl = DataLoader(train_ds, shuffle=True,   batch_size = batch_size, num_workers = num_workers, drop_last = True)
    valid_dl = DataLoader(valid_ds, shuffle=False,  batch_size = batch_size, num_workers = num_workers, drop_last = True)
    test_dl  = DataLoader(test_ds, shuffle=False,    batch_size = batch_size, num_workers = num_workers, drop_last = True)

    return train_dl, valid_dl, test_dl