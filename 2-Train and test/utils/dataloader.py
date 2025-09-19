import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import SequentialSampler, SubsetRandomSampler
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from PIL import Image
from pathlib import Path
from itertools import repeat, chain
import pickle


class RD_Dataset(Dataset):
    def __init__(self, dataset_path, transforms=None):
        self.dataset_path = dataset_path
        self.transforms = transforms
        
        self.dataset = []
        self.images = []
        i=0
        for filename in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, filename)
            seq_name = filename.split('.')[0]
            print('Loading dataset: ', seq_name)
            images_dir = os.path.join(Path(dataset_path).parent, seq_name)
      
            self.dataset.append(np.load(file_path, allow_pickle=True))
            self.dataset[i]['image_dir'] = images_dir
            i += 1
            #if i>0: break  # for debug 

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):        
        return index


def dataset_collate(batch):
    t_maps = []
    t_1_maps = []
    t_2_maps = []
    t_labels = []
    t_1_labels = []
    t_2_lables = []
    levels = []
    for maps, label, level in batch:
        t_maps.append(maps[0])
        t_1_maps.append(maps[1])
        t_2_maps.append(maps[2])
        t_labels.append(label[0])
        t_1_labels.append(label[1])
        t_2_lables.append(label[2])
        levels.append(level)

    maps = torch.from_numpy(np.array([t_maps, t_1_maps, t_2_maps], dtype=np.float64)).unsqueeze(2).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array([t_labels, t_1_labels, t_2_lables], dtype=np.float64)).unsqueeze(2).type(torch.FloatTensor)
  
    return maps, labels, levels


def get_dataloader(data_path, batch_size=8, use_dist=False):
    with open(os.path.join(data_path, "trainset.pkl"), 'rb') as f:
        train_dataset = pickle.load(f)
    with open(os.path.join(data_path, "valset.pkl"), 'rb') as f:
        val_dataset = pickle.load(f)
    with open(os.path.join(data_path, "testset.pkl"), 'rb') as f:
        test_dataset = pickle.load(f)
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
    train_idx = np.arange(len(train_dataset))
    val_idx = np.arange(len(val_dataset))
    test_idx = np.arange(len(test_dataset))

    if use_dist:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=True)
    else:
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SequentialSampler(test_idx)
    trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler, collate_fn=dataset_collate)
    testloader = DataLoader(dataset=test_dataset, batch_size=1, sampler=test_sampler, collate_fn=dataset_collate)
    validloader = DataLoader(dataset=val_dataset, batch_size=batch_size, sampler=val_sampler, collate_fn=dataset_collate)
    
    test_images = np.load(os.path.join(data_path, 'test_image_list.npy'))
    
    return trainloader, testloader, validloader, test_images




def dataset_collate_test(batch):
    t_maps = []
    t_1_maps = []
    t_2_maps = []
    t_labels = []
    for maps, label in batch:
        t_maps.append(maps[0])
        t_1_maps.append(maps[1])
        t_2_maps.append(maps[2])
        t_labels.append(label)

    maps = torch.from_numpy(np.array([t_maps, t_1_maps, t_2_maps], dtype=np.float64)).unsqueeze(2).type(torch.FloatTensor)
    labels = torch.from_numpy(np.array([t_labels], dtype=np.float64)).type(torch.FloatTensor)

    return maps, labels

    
def get_test_dataloader(gt_path, inft_path):
    gt = np.load(gt_path)
    intf = np.load(inft_path)
    #gt=abs(gt)
    #intf=abs(intf)
    min_i = min(gt.min(), intf.min())
    max_i = max(gt.max(), intf.max())
    gt = (gt-min_i) / (max_i-min_i)
    intf = (intf-min_i) / (max_i-min_i)
    dataset=[]
    dataset.append(np.repeat(intf[np.newaxis, :, :], 3, axis=0))
    dataset.append(gt)
    total_data = []
    total_data.append(dataset)
    testloader = DataLoader(dataset=total_data, batch_size=1, collate_fn=dataset_collate_test)

    return testloader, testloader, testloader

def dataset_collate_real_test(batch):
    maps = []
    for map in batch:
        maps.append(np.expand_dims(map[0,:,:], axis=0))
        maps.append(np.expand_dims(map[1,:,:], axis=0))
        maps.append(np.expand_dims(map[2,:,:], axis=0))
        maps = torch.from_numpy(np.stack(maps, dtype=np.float64)).unsqueeze(2).type(torch.FloatTensor)
    return maps

def get_test_real_dataloader(data_path):
    filenames = sorted(os.listdir(data_path))
    test_data = []
    for i in range(len(filenames)-2):
        frames = []
        frames.append(np.load(os.path.join(data_path,filenames[i+2]), allow_pickle=True))
        frames.append(np.load(os.path.join(data_path,filenames[i+1]), allow_pickle=True))
        frames.append(np.load(os.path.join(data_path,filenames[i]), allow_pickle=True))
        frames = np.stack(frames)
        test_data.append(frames)
    test_data = np.stack(test_data)
    min_value = np.min(test_data)   # -8.503986
    max_value = np.max(test_data)   # 121.01468
    test_data = (test_data-min_value) / (max_value-min_value)
    testloader = DataLoader(dataset=test_data, batch_size=1, collate_fn=dataset_collate_real_test)

    return testloader, None, None
    
    