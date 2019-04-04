import numpy as np
import pandas as pd
from sklearn import preprocessing
from collections import deque
from sklearn import preprocessing
from random import shuffle
from config import *

def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.

    :param df: pandas.DataFrame
    :param n:
    :return: pandas.DataFrame
    """
    import pdb; pdb.set_trace()
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df

def true_strength_index(df, r, s):
    """Calculate True Strength Index (TSI) for given data.

    :param df: pandas.DataFrame
    :param r:
    :param s:
    :return: pandas.DataFrame
    """
    M = pd.Series(df.diff(1))
    aM = abs(M)
    EMA1 = pd.Series(M.ewm(span=r, min_periods=r).mean())
    aEMA1 = pd.Series(aM.ewm(span=r, min_periods=r).mean())
    EMA2 = pd.Series(EMA1.ewm(span=s, min_periods=s).mean())
    aEMA2 = pd.Series(aEMA1.ewm(span=s, min_periods=s).mean())
    TSI = pd.Series(EMA2 / aEMA2)
    return TSI


MA_WINDOW = 20
def preprocess_df(df):
    df = df.drop('future', 1)

    targets = df['target']
    df = df.drop(['target'], axis=1)

    # https://github.com/Crypto-toolbox/pandas-technical-indicators
    df['BTC-TSI'] = true_strength_index(df['BTC-USD_close'], 25, 13)
    # import pdb
    # pdb.set_trace()
    df['BTC-MA'] = df['BTC-USD_close'].rolling(window=MA_WINDOW).mean()  # moving average
    df['BTC-EMA'] = df['BTC-USD_close'].ewm(span=MA_WINDOW).mean()  # moving average
    df['BTC-MOMENTUM'] = df['BTC-USD_close'].diff(MA_WINDOW)
    df['target'] = targets
    df = df[38:] # r + s

    for col in df.columns:
        print(col)
        if col != "target":
            df = df[df[col] != 0]
            if col != 'BTC-MOMENTUM' and col != 'BTC-TSI':
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
