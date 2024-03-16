from torch.utils.data import random_split
import pytorch_lightning as pl
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import ToTensor, Compose
import pandas as pd
import torch
from tqdm import tqdm

def to_numpy(feature):
    output = []
    for element in feature:
        output.append(element)
    output = np.array(output, dtype=np.float32)
    output = output.flatten()
    return output

def delete_nan(x):
    output = []
    for element in tqdm(x):
        if np.isnan(np.min(x)) == False:
            output.append(element)
            print(element)
    output = np.array(output)
    return output

class Dataset(Dataset):
    def __init__(self):
        print("Loading Dataset")
        lightning = pd.read_pickle('cleanedData/lightning.pkl').to_numpy()
        population = pd.read_pickle('cleanedData/population.pkl').to_numpy()
        rain = pd.read_pickle('cleanedData/rain.pkl').to_numpy()
        biomass = pd.read_pickle('cleanedData/biomass.pkl').to_numpy()
        temperature = pd.read_pickle('cleanedData/temperature.pkl').to_numpy()
        humidity = pd.read_pickle('cleanedData/humidity.pkl').to_numpy()
        wind = pd.read_pickle('cleanedData/wind.pkl').to_numpy()
        fireCCIL1982_2018 = pd.read_pickle('cleanedData/fireCCIL1982-2018.pkl').to_numpy()

        lightning = to_numpy(lightning)
        population = to_numpy(population)
        rain = to_numpy(rain)
        biomass = to_numpy(biomass)
        temperature = to_numpy(temperature)
        humidity = to_numpy(humidity)
        wind = to_numpy(wind)
        fireCCIL1982_2018 = to_numpy(fireCCIL1982_2018)

        self.y = fireCCIL1982_2018

        self.x = []
        self.x.append(lightning)
        self.x.append(population)
        self.x.append(rain)
        self.x.append(biomass)
        self.x.append(humidity)
        self.x.append(wind)
        self.x.append(temperature)
        self.x = np.array(self.x, dtype=np.float32)
        self.x = self.x.transpose()
        data = np.concatenate((self.x, self.y.reshape(-1,1)),axis=1)
        data = data[~np.isnan(data).any(axis=1), :]
        self.x = data[:, 0:7]
        self.y = data[:, 7]
        self.x = torch.from_numpy(self.x)

        self.y = torch.from_numpy(self.y)

        self.length = len(self.x)

        print(f"X shape: {self.x.shape}")
        print(f"Y shape: {self.y.shape}")
        print(f"Length: {self.length}")
        print("Finished Loading Dataset")
        
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def main():
    dataset = Dataset()

if __name__=='__main__':
    main()
