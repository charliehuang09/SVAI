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
    path='../data/burned_area_data/FireCCILT11-1982-2018_T62.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.BA.values
    dataset = np.flip(dataset, axis=1)
    print(dataset.shape)
    with open('../cleanedData/forcing_data/FireCCIL1982-2018.npy', 'wb') as f:
        np.save(f, dataset)

    with open('../cleanedData/forcing_data/FireCCIL1982-2018Index.npy', 'wb') as f:
            np.save(f, index)


if __name__=='__main__':
    main()
