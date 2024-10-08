import xarray as xr
from datetime import datetime
import numpy as np
from datetime import datetime
import cv2
from misc import *
import pandas as pd


def resize(x):
    x = cv2.resize(x, (144, 96))
    return x


def main():
    path = '../data/forcing_data/wind.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.WIND.values
    dataset = np.flip(dataset, axis=1)
    dataset = to_africa(dataset)

    dataset = dataset.tolist()

    index = to_datetime(index)
    index = pd.DatetimeIndex(index)

    df = pd.Series(dataset, index=index)

    df = df.resample('ME').bfill()

    df = df[df.index > datetime.datetime(year=2001, month=1, day=1)]
    df = df[df.index < datetime.datetime(year=2011, month=1, day=1)]

    df = scale(df)
    assert np.nanmax(np.array(df.tolist(
    ))) == 1, f"Max after rescaling is {np.nanmax(np.array(df.tolist()))}"

    df.to_pickle("../cleanedData/wind.pkl")


if __name__ == '__main__':
    main()
