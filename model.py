import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Metric
import torchvision
from torchvision.transforms import CenterCrop
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.input = nn.Linear(6, 24)
        self.fc1 = nn.Linear(24, 24)
        self.fc2 = nn.Linear(24, 24)
        self.output = nn.Linear(24, 1)
    
    def forward(self, x):
        x = self.input(x)
        
        x = self.fc1(x)
        x = self.fc2(x)

        x = self.output(x)

        return x

def main():
    model = Model()

if __name__=='__main__':
    main()
