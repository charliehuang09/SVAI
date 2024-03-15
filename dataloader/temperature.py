import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from dateutil.parser import parse
from datetime import datetime, timedelta
import cv2
import pandas as pd
from misc import *

def main():
    path='../data/forcing_data/temperature.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.TBOT.values
    dataset = np.flip(dataset, axis=1)

    dataset = dataset.tolist()

    index = to_datetime(index)
    index = pd.DatetimeIndex(index)

    df = pd.Series(dataset, index=index)

    df = df.resample('M').bfill()

    df = df[df.index > datetime.datetime(year=2001, month=1, day=1)]
    df = df[df.index < datetime.datetime(year=2010, month=1, day=1)]

    df.to_pickle("../cleanedData/temperature.pkl")


if __name__=='__main__':
    main()
