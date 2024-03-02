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
    path='../data/forcing_data/lightning_1995-2011.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.lnfm.values
    dataset = np.flip(dataset, axis=1)
    output = []
    for idx, _ in enumerate(dataset):
        output.append(resize(dataset[idx]))
    dataset = np.array(output)
    print(dataset.shape)
    with open('../cleanedData/forcing_data/lightning.npy', 'wb') as f:
        np.save(f, dataset)

    with open('../cleanedData/forcing_data/lightningIndex.npy', 'wb') as f:
        np.save(f, index)


if __name__=='__main__':
    main()
