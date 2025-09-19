# used for spliting total dataset, save to 'train' 'test' and 'valid', [N, 3, 64, 128]

from torch.utils.data import Dataset
import numpy as np
import os
from pathlib import Path
from itertools import repeat, chain


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
    

def main(data_path, train_ratio=0.8):
    RD_dataset = RD_Dataset(dataset_path=data_path)
    total_dataset = []
    gt = []
    datalen = 0
    dataset = []
    level = 7 
    image_list = []  # interference power=[-5:5:25]
    # frame_dataset = np.zeros([3, 128, 64], dtype=np.float64)
    # total frame samples = 384685
    for i in range(len(RD_dataset)):
        total_dataset = RD_dataset.dataset[i]['sb'].transpose(2,0,1)
        gt = RD_dataset.dataset[i]['sb0'].transpose(2,0,1)   
        
        for j in range(gt.__len__()-2*level):
            list_data = []
            frame_t = total_dataset[j+2*level,:,:].reshape(1,64,128)
            assert not np.all(frame_t==0)
            frame_dataset = np.concatenate([total_dataset[j+2*level,:,:].reshape(1,64,128), \
                                            total_dataset[j+level,:,:].reshape(1,64,128), \
                                            total_dataset[j,:,:].reshape(1,64,128)], axis=0)
            list_data.append(frame_dataset)
            gts = np.concatenate([gt[j+2*level,:,:].reshape(1,64,128), \
                                  gt[j+level,:,:].reshape(1,64,128), \
                                  gt[j,:,:].reshape(1,64,128)],axis=0)
            list_data.append(gts)
            sinr_level = j%level
            list_data.append(sinr_level)
            dataset.append(list_data)
        image_dir = RD_dataset.dataset[i]['image_dir']
        image_list_sorted = sorted(os.listdir(image_dir))
        image_list_aug = list(chain.from_iterable(repeat((image_dir + '/' + item), 7)  for item in image_list_sorted))
        image_list.append(image_list_aug[14:])        
    datalen = len(dataset)
    image_list = np.concatenate(image_list)
    dataidx = np.array(list(range(datalen)))
    np.random.shuffle(dataidx)

    splitfrac = train_ratio
    split_idx = int(splitfrac * (datalen))  
    train_idxs = dataidx[:split_idx]
    valid_idxs = dataidx[split_idx:]

    testsplit = 0.1
    testidxs = int(testsplit * len(train_idxs))

    test_idxs = train_idxs[:testidxs]
    train_idxs = train_idxs[testidxs:] # number of train samples: 318519
    test_images = image_list[test_idxs]
    np.save(os.path.join('/home/liululu/dataset/radical/data_split/', "test_image_list.npy"), test_images)
    np.save(os.path.join('/home/liululu/dataset/radical/data_split/', "trainset.npy"), [dataset[i] for i in train_idxs])
    np.save(os.path.join('/home/liululu/dataset/radical/data_split/', "valset.npy"), [dataset[i] for i in valid_idxs])
    np.save(os.path.join('/home/liululu/dataset/radical/data_split/', "testset.npy"), [dataset[i] for i in test_idxs])
    


if __name__ == '__main__':
    data_path = ''
    main(data_path)
