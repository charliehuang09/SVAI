import xarray as xr
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from dateutil.parser import parse
import cv2
import pandas as pd
from misc import *

def to_datetime(index):
    output = []
    for element in index:
        output.append(element.astype(datetime.datetime))
    output = np.array(output)
    return output

def remove_zeros(dataset):
    print("Removing Zeros")
    print(f"Before: {np.nanmean(dataset)}")

    dataset = np.swapaxes(dataset, 0, 2)
    for i in range(len(dataset)):
        for j in range(len(dataset[i])):
            if (not np.any(dataset[i][j])):
                dataset[i][j] = np.nan
    dataset = np.swapaxes(dataset, 0, 2)
    
    print(f"Before: {np.nanmean(dataset)}")
    return dataset

def main():
    print("Loading fireCCIL")
    path='../data/burned_area_data/FireCCILT11-1982-2018_T62.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.BA.values
    dataset = np.flip(dataset, axis=1)
    dataset = to_africa(dataset)
    
    dataset = remove_zeros(dataset)
    dataset[dataset > 2e6] =  np.nan

    dataset = dataset.tolist()

    index = to_datetime(index)
    index = pd.DatetimeIndex(index)

    df = pd.Series(dataset, index=index)

    df = df.resample('M').bfill()

    df = df[df.index > datetime.datetime(year=2001, month=1, day=1)]
    df = df[df.index < datetime.datetime(year=2011, month=1, day=1)]

    df.to_pickle("../cleanedData/fireCCIL1982-2018.pkl")

    print("Finished Loading FireCCIL")


if __name__=='__main__':
    main()
