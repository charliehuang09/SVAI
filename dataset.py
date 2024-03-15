from torch.utils.data import random_split
import pytorch_lightning as pl
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
from torchvision.transforms import ToTensor, Compose
class CustomDataset(Dataset):
    def __init__(self):
        biomass = np.load("cleanedData/forcing_data/biomass.npy")
        humidity = np.load("cleanedData/forcing_data/humidity.npy")
        rain = np.load("cleanedData/forcing_data/rain.npy")
        population = np.load("cleanedData/forcing_data/population.npy")
        gfed2001_2015 = np.load("cleanedData/forcing_data/gfed2001-2015.npy")
        lightning = np.load("cleanedData/forcing_data/lightning.npy")
        temperature = np.load("cleanedData/forcing_data/temperature.npy")
        wind = np.load("cleanedData/forcing_data/wind.npy")
        fireCCIL1982_2018 = np.load("cleanedData/forcing_data/FireCCIL1982-2018.npy")

        biomassIndex = np.load("cleanedData/forcing_data/biomassIndex.npy", allow_pickle=True)
        humidityIndex = np.load("cleanedData/forcing_data/humidityIndex.npy", allow_pickle=True)
        rainIndex = np.load("cleanedData/forcing_data/rainIndex.npy", allow_pickle=True)
        populationIndex = np.load("cleanedData/forcing_data/populationIndex.npy", allow_pickle=True)
        gfed2001_2015Index = np.load("cleanedData/forcing_data/gfed2001-2015Index.npy", allow_pickle=True)
        lightningIndex = np.load("cleanedData/forcing_data/temperatureIndex.npy", allow_pickle=True)
        temperatureIndex = np.load("cleanedData/forcing_data/temperatureIndex.npy", allow_pickle=True)
        windIndex = np.load("cleanedData/forcing_data/windIndex.npy", allow_pickle=True)
        fireCCIL1982_2018Index = np.load("cleanedData/forcing_data/FireCCIL1982-2018Index.npy", allow_pickle=True)
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index].astype('float32') / 255, self.y[index].astype('float32') / 255