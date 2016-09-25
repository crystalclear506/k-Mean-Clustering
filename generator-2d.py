#! /usr/local/bin/python
import pandas as pd
import numpy as np

def twoDimensionalDataGen():
    arr = []
    for i in range(100):
        arr.append([np.random.normal(scale=100), np.random.normal(scale=100)])
        arr.append([500 + np.random.normal(scale=100), 500 + np.random.normal(scale=100)])
        arr.append([np.random.normal(scale=100), 500 + np.random.normal(scale=100)])
    return np.array(arr)


df = pd.DataFrame(twoDimensionalDataGen())
df.to_csv("test_data_2D.csv", index=False, header=None)