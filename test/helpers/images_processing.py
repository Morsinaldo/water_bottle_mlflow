"""
Creator: Morsinaldo Medeiros
Date: 06-02-2023
Description: This script contains the preprocessing classes.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    """Custom Dataset for loading images"""
    def __init__(self, X, y):
        self.x = X
        self.y = y
        self.n_samples = len(X)
        
    def __getitem__(self, index):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(self.x[index]), torch.tensor(self.y[index], dtype=torch.int64)
    
    def __len__(self):
        return self.n_samples