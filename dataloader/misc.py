from datetime import datetime
import numpy as np
import datetime
import cv2

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

def to_africa(data):
    data = data[:, 20:70, :]
    left = data[:, :, :20]
    right = data[:, :, 132:]
    data = np.dstack((right, left))
    return data