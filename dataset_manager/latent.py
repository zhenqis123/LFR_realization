import torch.nn
from torch.utils.data import Dataset
import glob
import os
import cv2
import numpy as np
from torchvision.io import read_image
from torchvision import transforms
class Latent(Dataset):

    def __init__(self, rolled_path, ridge_path, mask_path, info_path, numPerImg, transform=None):
        self.info_path = info_path
        self.rolled_path = rolled_path
        self.ridge_path = ridge_path
        self.mask_path = mask_path
        self.transform = transform
        self.numPerImg = numPerImg
        self.info = np.loadtxt(self.info_path, dtype=str, delimiter=',')

    def __getitem__(self, index):
        index = index // self.numPerImg
        rolled_name = os.path.join(self.rolled_path, self.info[index][0])
        ridge_name = os.path.join(self.ridge_path, self.info[index][1])
        mask_name = os.path.join(self.mask_path, self.info[index][2])
        rolled_img = cv2.imread(rolled_name, cv2.IMREAD_GRAYSCALE)
        ridge_img = cv2.imread(ridge_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255
        target = rolled_img
        
        latent_img = self.transform(rolled_img).float()
        
        target = transforms.ToTensor()(target).float()
        ridge_img = transforms.ToTensor()(ridge_img).float()
        mask = transforms.ToTensor()(mask).float()

        
        return latent_img, target, ridge_img, mask
        pass

    def __len__(self):
        return len(self.info) * self.numPerImg
        pass