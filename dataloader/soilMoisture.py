import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import pandas as pd
from misc import *
from dateutil.relativedelta import relativedelta


def to_numpy(input):
    output = []
    for element in input:
        output.append(element.data)
    output = np.array(output)
    return output.squeeze()


def to_datetime(input):
    output = []
    basetime = datetime.datetime(
        year=1960,
        month=1,
        day=1,
        hour=1,
        minute=1,
        second=1,
    )
    for time in input:
        output.append(basetime +
                      relativedelta(months=time.data.tolist() - 0.5))
    output = np.array(output)
    return output


def main():
    print("Loading Soil Moisture")
    path = '../data/forcing_data/soil_moisture.nc'
    data = Dataset(path)
    index = data.variables['T']
    dataset = data.variables['moisture']
    dataset = to_numpy(dataset)
    dataset = resize(dataset)
    dataset = to_africa(dataset)
    dataset[dataset > 5e20] = np.nan

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

    df.to_pickle("../cleanedData/soil_moisture.pkl")

    print("Finished Loading Soil Moisture")


if __name__ == '__main__':
    main()
