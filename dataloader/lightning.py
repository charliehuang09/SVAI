import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from dateutil.parser import parser
import datetime
import cv2
import datetime
import time
import pandas as pd
import datetime
from tqdm import trange
from misc import *

startYear = 2000
endYear = 2011

def change_year(index, year):
    for i in range(len(index)):
        index[i] = index[i].replace(year=year)
    return index

def duplicate_index(index):
    output = []
    for year in range(startYear, endYear):
        output.append(change_year(index, year))
    output = tuple(output)
    output = np.array(np.concatenate(output))
    return output

def duplicate_dataset(dataset):
    dataset = np.repeat(dataset, endYear-startYear, axis=0)
    return dataset

def convert_numpy(input):
    output = []
    for element in input:
        output.append(element)
    output = np.array(output)
    return output

def mean(input):
    input = convert_numpy(input)
    input = np.nanmean(input, axis=0)
    return input

def main():
    path='../data/forcing_data/lightning_1995-2011.nc'
    data = xr.open_dataset(path)
    dataset = data.lnfm.values
    dataset = np.flip(dataset, axis=1)
    dataset = resize(dataset)
    dataset = dataset.tolist()

    dataframes = []
    for year in trange(startYear, endYear):
        index = data.time.values
        index = change_year(index, year)#dummy year
        index = to_datetime(index)
        index = pd.DatetimeIndex(index)

        df = pd.Series(dataset, index=index)
        df = df.resample('M').apply(mean)
        dataframes.append(df)
    
    df = pd.concat(dataframes)

    df = df[df.index > datetime.datetime(year=2001, month=1, day=1)]
    df = df[df.index < datetime.datetime(year=2010, month=1, day=1)]

    df.to_pickle("../cleanedData/lightning.pkl")

if __name__=='__main__':
    main()
