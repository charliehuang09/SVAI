import xarray as xr
import datetime
import numpy as np
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
    print("Loading MCD64")
    path = '../data/burned_area_data/MCD64CMQ-2001-2019_T62.nc'
    data = xr.open_dataset(path)
    index = data.time.values
    dataset = data.BA.values
    dataset = np.flip(dataset, axis=1)
    dataset = to_africa(dataset)

    dataset = remove_zeros(dataset)

    dataset[dataset <= 0.5] = 0
    dataset[dataset > 0.5] = 1

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

    df.to_pickle("../cleanedData/MCD64.pkl")

    print("Finished Loading MCD64")


if __name__ == '__main__':
    main()
