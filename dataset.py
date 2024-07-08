from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Literal

def to_numpy(feature):
    output = []
    for element in feature:
        output.append(element)
    output = np.array(output, dtype=np.float32)
    return output

def delete_nan(x):
    output = []
    for element in tqdm(x):
        if np.isnan(np.min(x)) == False:
            output.append(element)
    output = np.array(output)
    return output

def scale(x):
    min = np.nanmin(x)
    max = np.nanmax(x)
    x = ((x - min) / (max - min))
    return x

class Dataset(Dataset):
    def __init__(self, type: Literal["Train", "Valid"], train_test_split, modelType : Literal['Regression', 'Classification'], shift=False, verbose=True):
        if (type == 'Train'):
            years = (0, int(120 * train_test_split))
        if (type == 'Valid'):
            years = (int(120 * train_test_split), 120)
        if (verbose):
            print(f"Range: {years[0]}-{years[1]}")
            print(f"Loading {type} Dataset")
        lightning = pd.read_pickle('cleanedData/lightning.pkl').iloc[years[0]:years[1]].to_numpy()
        population = pd.read_pickle('cleanedData/population.pkl').iloc[years[0]:years[1]].to_numpy()
        rain = pd.read_pickle('cleanedData/rain.pkl').iloc[years[0]:years[1]].to_numpy()
        biomass = pd.read_pickle('cleanedData/biomass.pkl').iloc[years[0]:years[1]].to_numpy()
        temperature = pd.read_pickle('cleanedData/temperature.pkl').iloc[years[0]:years[1]].to_numpy()
        humidity = pd.read_pickle('cleanedData/humidity.pkl').iloc[years[0]:years[1]].to_numpy()
        wind = pd.read_pickle('cleanedData/wind.pkl').iloc[years[0]:years[1]].to_numpy()
        soil_moisture = pd.read_pickle('cleanedData/soil_moisture.pkl').iloc[years[0]:years[1]].to_numpy()
        fireCCIL1982_2018 = pd.read_pickle('cleanedData/fireCCIL1982-2018.pkl').iloc[years[0]:years[1]].to_numpy()
        mcd64 = pd.read_pickle('cleanedData/MCD64.pkl').iloc[years[0]:years[1]].to_numpy()

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

        if (modelType == 'Regression'):
            self.y = fireCCIL1982_2018 # Regression
        if (modelType == 'Classification'):
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
        
        if (shift):
            data = np.concatenate((self.y[np.newaxis, :], self.x), axis=0)
            self.x = np.concatenate((np.zeros((9, 1, 50, 32)), data), axis=1)
            self.y = np.concatenate((data, np.zeros((9, 1, 50, 32))), axis=1)
            
            self.x = self.x.astype(np.float32)
            self.y = self.y.astype(np.float32)
            
            if (verbose):
                print(self.x.shape)
                print(self.y.shape)
            
        self.x = self.x.reshape(9, -1)
        self.x = self.x.transpose()
        
        self.y = self.y.reshape(9, -1)
        self.y = self.y.transpose()
        
        data = np.concatenate((self.x, self.y), axis=1)
        data = data[~np.isnan(data).any(axis=1), :]
        
        self.x = data[:, :9]
        self.y = data[:, 9:]
        
        self.xmin = []
        self.xmax = []
        for element in self.x.transpose():
            self.xmin.append(np.nanmin(element))
            self.xmax.append(np.nanmax(element))
        
        for i in range(9):
            self.x[:, i] = scale(self.x[:, i])
            self.y[:, i] = scale(self.y[:, i])

        if (modelType == 'Regression'):
            self.y[:, 0] = self.y[:, 0] / self.y[:, 0].mean()
            pass
        if (modelType == 'Classification'):
            pass
        
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)

        assert len(self.y) == len(self.x)
        self.length = len(self.x)
        
        if (verbose):
            print(f"X shape: {self.x.shape}")
            print(f"Y shape: {self.y.shape}")
            print(f"Length: {self.length}")
            print("Finished Loading Dataset\n")
    
    def getx(self):
        return self.x
    def gety(self):
        return self.y
    def getxmin(self):
        return self.xmin
    def getxmax(self):
        return self.xmax
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def main():
    train_dataset = Dataset("Train", 0.8, 'Regression', True)
    valid_dataset = Dataset("Valid", 0.8, 'Classification', True)
    
    for batch in train_dataset:
        x, y = batch

if __name__=='__main__':
    main()
