import xarray as xr
from datetime import datetime
import numpy as np
from misc import *
import pandas as pd

def main():
    path='../data/forcing_data/humidity.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.RH2M.values
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
    assert np.nanmax(np.array(df.tolist())) == 1, f"Max after rescaling is {np.nanmax(np.array(df.tolist()))}"
    
    df.to_pickle("../cleanedData/humidity.pkl")

if __name__=='__main__':
    main()
