import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
from sklearn import preprocessing
from random import shuffle
from config import *


def preprocess_df(df):
    df = df.drop('future', 1)
    for col in df.columns:
        print(col)
        if col != "target":
            df = df[df[col] != 0]
            df[col] = df[col].pct_change()  # normalize
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1
    df.dropna(inplace=True)

    sequential_data = []
    prev_days = deque(maxlen=SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])  # dont take target
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    shuffle(sequential_data)
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    shuffle(buys)
    shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    shuffle(sequential_data)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y
