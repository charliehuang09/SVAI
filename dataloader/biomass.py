import xarray as xr
from datetime import datetime
import numpy as np
from datetime import datetime
from misc import *
import pandas as pd

def main():
    print("Loading Biomass")
    path='../data/forcing_data/biomass.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.biomass.values
    dataset[np.isnan(dataset)] = np.nanmean(dataset)
    dataset = resize(dataset)
    dataset = np.dstack((dataset[:, :, 72:], dataset[:, :, :72]))
    dataset = to_africa(dataset)

    dataset = dataset.tolist()

    index = to_datetime(index)
    index = pd.DatetimeIndex(index)

    df = pd.Series(dataset, index=index)

    df = df.resample('ME').bfill()

    df = df[df.index > datetime.datetime(year=2001, month=1, day=1)]
    df = df[df.index < datetime.datetime(year=2011, month=1, day=1)]

    df.to_pickle("../cleanedData/biomass.pkl")

    print("Finished Loading Biomass")

if __name__=='__main__':
    main()
