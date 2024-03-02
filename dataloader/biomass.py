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
    path='../data/forcing_data/biomass.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.biomass.values
    dataset[np.isnan(dataset)] = np.nanmean(dataset)
    output = []
    for idx, _ in enumerate(dataset):
        output.append(resize(dataset[idx]))
    dataset = np.array(output)
    dataset = np.dstack((dataset[:, :, 72:], dataset[:, :, :72]))

    with open('../cleanedData/forcing_data/biomass.npy', 'wb') as f:
        np.save(f, dataset)

    with open('../cleanedData/forcing_data/biomassIndex.npy', 'wb') as f:
        np.save(f, index)

if __name__=='__main__':
    main()
