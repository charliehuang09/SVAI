import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from dateutil.parser import parse
from datetime import datetime, timedelta
import cv2

def resize(x):
    x = cv2.resize(x, (144, 96)) 
    return x

def main():
    path='../data/forcing_data/population.nc'
    data = xr.open_dataset(path)
    print(data)
    dataset = data.hdm.values
    output = []
    for idx, _ in enumerate(dataset):
        output.append(resize(dataset[idx]))
    dataset = np.array(output)
    print(dataset.shape)
    dataset = np.flip(dataset, axis=1)
    with open('../cleanedData/forcing_data/population.npy', 'wb') as f:
        np.save(f, dataset)

if __name__=='__main__':
    main()
