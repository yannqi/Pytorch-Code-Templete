import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
from PIL import Image



class CatVsDogDataset(Dataset):
    def __init__(self, annotations_file, img_dir, is_train=True ,transform=None, target_transform=None):
        if is_train:
            annotations_file = annotations_file + '/train.txt'
            img_dir = img_dir + 'train'
        else:
            annotations_file = annotations_file + '/test1.txt'
            img_dir = img_dir + 'train'
            
        f = open(annotations_file)
        self.img_labels  = f.read().split()
        f.close()
        
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx])
        img = Image.open(img_path)
        label_name = self.img_labels[idx][:3]
        if label_name == 'cat':
            label = torch.tensor(0)
        elif label_name == 'dog': label = torch.tensor(1)
        else :label = torch.tensor(-1)
        if self.transform:
            img = self.transform(img)
              
        if self.target_transform:
            label = self.target_transform(label)
        return img, label, img_path