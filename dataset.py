from torch.utils.data import random_split
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import normalize
from typing import Literal
from config import ModelType

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

datasetType = Literal["Train", "Valid"]
class Dataset(Dataset):
    def __init__(self, type: datasetType, train_test_split, modelType):
        print(f"Loading {type} Dataset")
        lightning = pd.read_pickle('cleanedData/lightning.pkl').to_numpy()
        population = pd.read_pickle('cleanedData/population.pkl').to_numpy()
        rain = pd.read_pickle('cleanedData/rain.pkl').to_numpy()
        biomass = pd.read_pickle('cleanedData/biomass.pkl').to_numpy()
        temperature = pd.read_pickle('cleanedData/temperature.pkl').to_numpy()
        humidity = pd.read_pickle('cleanedData/humidity.pkl').to_numpy()
        wind = pd.read_pickle('cleanedData/wind.pkl').to_numpy()
        soil_moisture = pd.read_pickle('cleanedData/soil_moisture.pkl').to_numpy()
        fireCCIL1982_2018 = pd.read_pickle('cleanedData/fireCCIL1982-2018.pkl').to_numpy()
        mcd64 = pd.read_pickle('cleanedData/MCD64.pkl')

        lightning = to_numpy(lightning)
        population = to_numpy(population)
        rain = to_numpy(rain)
        biomass = to_numpy(biomass)
        temperature = to_numpy(temperature)
        humidity = to_numpy(humidity)
        wind = to_numpy(wind)
        soil_moisture = to_numpy(soil_moisture)
        fireCCIL1982_2018 = to_numpy(fireCCIL1982_2018)
        mcd64 = to_numpy(mcd64)

        if (modelType == ModelType.Regression):
            self.y = fireCCIL1982_2018 # Regression
        if (modelType == ModelType.Classification):
            self.y = mcd64 #Classification

        self.x = []
        self.x.append(lightning)
        self.x.append(population)
        self.x.append(rain)
        self.x.append(biomass)
        self.x.append(humidity)
        self.x.append(wind)
        self.x.append(soil_moisture)
        self.x.append(temperature)
        self.x = np.array(self.x, dtype=np.float32)
        self.x = self.x.transpose()

        data = np.concatenate((self.x, self.y.reshape(-1,1)), axis=1)
        data = data[~np.isnan(data).any(axis=1), :]
        
        self.x = normalize(data[:, 0:8])
        self.y = data[:, 8]
        
        if (modelType == ModelType.Regression):
            self.y = self.y/self.y.mean()
        if (modelType == ModelType.Classification):
            pass
        

        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

        if (type == "Train"):
            print(f"Train Range: 0-{int(len(self.x) * train_test_split)}")
            self.x = self.x[:int(len(self.x) * train_test_split)]
            self.y = self.y[:int(len(self.y) * train_test_split)]

        if (type == "Valid"):
            print(f"Train Range: {int(len(self.x) * train_test_split)}-{len(self.x)}")
            self.x = self.x[int(len(self.x) * train_test_split):]
            self.y = self.y[int(len(self.y) * train_test_split):]

        self.length = len(self.x)

        print(f"X shape: {self.x.shape}")
        print(f"Y shape: {self.y.shape}")
        print(f"Length: {self.length}")
        print("Finished Loading Dataset\n")
        
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def main():
    train_dataset = Dataset("Train", 0.8, ModelType.Regression)
    valid_dataset = Dataset("Valid", 0.8, ModelType.Classification)

if __name__=='__main__':
    main()
