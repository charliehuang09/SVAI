from datetime import datetime
import numpy as np
from dateutil.parser import parse
import datetime
import cv2
import pandas as pd

def to_datetime(input):
    output = []
    for time in input:
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minute = time.minute
        second = time.second
        output.append(datetime.datetime(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
        ))
    return output


def resize(input):
    output = []
    for idx, _ in enumerate(input):
        output.append(cv2.resize(input[idx], (144, 96)))
    output = np.array(output)
    return output