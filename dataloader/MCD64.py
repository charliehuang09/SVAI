import xarray as xr
import matplotlib.pyplot as plt
import datetime
import os
import numpy as np
from dateutil.parser import parse
import cv2
import pandas as pd

def to_datetime(index):
    output = []
    for element in index:
        output.append(element.astype(datetime.datetime))
    output = np.array(output)
    return output

def main():
    path='../data/burned_area_data/MCD64CMQ-2001-2019_T62.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.BA.values
    print(dataset.shape)
    dataset = np.flip(dataset, axis=1)

    dataset = dataset.tolist()

    index = to_datetime(index)
    index = pd.DatetimeIndex(index)

    df = pd.Series(dataset, index=index)

    df = df.resample('M').bfill()

    df = df[df.index > datetime.datetime(year=2001, month=1, day=1)]
    df = df[df.index < datetime.datetime(year=2010, month=1, day=1)]

    df.to_pickle("../cleanedData/MCD64.pkl")


if __name__=='__main__':
    main()
