from tensorflow.python.client import device_lib
import pandas as pd
import os
from sklearn import preprocessing
import numpy as np
import pandas as pd
from collections import deque
import numpy as np
import random
from preprocess import preprocess_df
from config import *
from utils import classify

import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from model import build_model
from tensorflow.keras.models import load_model



def train():
    main_df = pd.DataFrame()
    # ratios = ["BTC-USD", "LTC-USD", "ETH-USD"] #, "BCH-USD"]
    ratios = ["BTC-USD", "EOS-USD", "ETH-USD", "BCHABC-USD", "LTC-USD"] #, "BCH-USD"]
    for ratio in ratios:
        dataset = f"crypto_data/{ratio}.csv"
        df = pd.read_csv(dataset, names=["time", "low", "high", "open", "close", "volume"])

        df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)
        df.set_index("time", inplace=True)
        df = df[[f"{ratio}_close", f"{ratio}_volume"]]
        # print(df.head())
        if len(main_df) == 0:
            main_df = df
        else:
            main_df = main_df.join(df)
    # main_df["BTC-MOMENTUM"] = df["BTC-MOMENTUM"]
    # main_df["BTC-MA"] = df["BTC-MA"]
    # main_df["BTC-EMA"] = df["BTC-EMA"]

    main_df['future'] = main_df[f"{RATIO_TO_PREDICT}_close"].shift(-FUTURE_PERIOD_PREDICT)
    main_df = main_df[:-FUTURE_PERIOD_PREDICT]

    print(main_df[[f"{RATIO_TO_PREDICT}_close", "future"]].head())
    main_df["target"] = list(map(classify, main_df[f"{RATIO_TO_PREDICT}_close"], main_df["future"]))


    times = sorted(main_df.index.values)
    last_5pct = times[-int(0.05*len(times))]

    validation_main_df = main_df[(main_df.index >= last_5pct)]
    main_df = main_df[(main_df.index < last_5pct)]


    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)
    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model
    opt = tf.keras.optimizers.Adam(lr=LEARNING_RATE) #, decay=DECAY)
    model = build_model(LOSS, opt)


    # model = load_model(
    #     "models/RNN_Final-03-0.588.model",
    #     custom_objects=None,
    #     compile=True
    # )
    print(model.summary())
    # import pdb; pdb.set_trace()
    tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
    filepath = "RNN_Final-{epoch:02d}-{val_acc:.3f}"  # unique file name that will include the epoch and the validation acc for that epoch
    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')) # saves only the best ones
    # Train model
    history = model.fit(
        train_x, train_y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(validation_x, validation_y),
        callbacks=[tensorboard, checkpoint],
    )

    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(NAME))

def predict(model, observation):
    out = model.predict(observation)
    return out


from collections import deque
from binance.client import Client
from binance_config import api_key, api_secret
import binance_constants
from config import currencies
from preprocess import true_strength_index


def get_live_data():
    client = Client(api_key, api_secret)
    feed = deque()
    main_data = pd.DataFrame()
    for currency in currencies:
        data = client.get_historical_klines(
            symbol=currency, interval=binance_constants.KLINE_INTERVAL_5MINUTE,
            start_str="5 days ago UTC")
        time = list(map(lambda x: [float(x[OPEN_TIME_IDX])], data))[:-1] # opentime open high low close volume  DROP LAST ONE??? GRABAGE?
        data = list(map(lambda x: [float(x[CLOSE_IDX]), float(x[VOLUME_IDX])], data))[:-1]  # opentime open high low close volume

        print(currency)
        print("data.head()", data[:2], time[:2])
        print("data.tail()", data[-2:], time[-2:])

        data = pd.DataFrame(data, columns=[f"{currency}_close", f"{currency}_volume"])
        main_data = pd.concat([main_data, data], axis=1)


    MA_WINDOW = 20
    df = main_data
    df['BTC-TSI'] = true_strength_index(df['BTCUSDT_close'], 25, 13)
    # import pdb
    # pdb.set_trace()
    df['BTC-MA'] = df['BTCUSDT_close'].rolling(window=MA_WINDOW).mean()  # moving average
    df['BTC-EMA'] = df['BTCUSDT_close'].ewm(span=MA_WINDOW).mean()  # moving average
    df['BTC-MOMENTUM'] = df['BTCUSDT_close'].diff(MA_WINDOW)
    df = df[38:]

    for col in df.columns:
        df = df[df[col] != 0]
        if col != 'BTC-MOMENTUM' and col != 'BTC-TSI':
            df[col] = df[col].pct_change()  # normalize
        df.dropna(inplace=True)
        df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1

    df.dropna(inplace=True)
    # df = df[-SEQ_LEN:]
    return np.array([df.values]) # test data


def test():
    model1 = load_model(
        # 420 min 120 min prediction
        "models/RNN_Final-14-0.571.model",
        custom_objects=None,
        compile=True
    )
    model2 = load_model(
        # 420 min 120 min prediction
        "models/RNN_Final-03-0.559.model",
        custom_objects=None,
        compile=True
    )
    model3 = load_model(
        # 420 min 120 min prediction
        "models/RNN_Final-16-0.575.model",
        custom_objects=None,
        compile=True
    )
    model4 = load_model(
        # 420 min 120 min prediction
        "models/RNN_Final-07-0.546.model",
        custom_objects=None,
        compile=True
    )


    data = get_live_data()
    recent = data[:,-SEQ_LEN:]

    out = model1.predict(recent)
    print(out)
    out = model2.predict(recent)
    print(out)
    out = model3.predict(recent)
    print(out)
    out = model4.predict(recent)
    print(out)


def run():

    model = load_model(
        # "models/RNN_Final-21-0.562.model",
        # 420 min 120 min prediction
        "RNN_Final-07-0.669.model",
        custom_objects=None,
        compile=True
    )
    import time
    import datetime

    funds = 100
    quant = 10
    buy_streak = 0
    sell_streak = 0
    streak = 5
    client = Client(api_key, api_secret)
    in_pos = 0 # sell
    actions = []
    prices = []
    # while True:
    for i in range(1):
        # time.sleep(60)
        output = go_live()
        action = output.argmax()
        price = client.get_symbol_ticker(symbol="BTCUSDT")
        prices += [price['price']]
        actions += [action]
        if action == 0:
            sell_streak += 1
            buy_streak = 0
        else:
            buy_streak += 1
            sell_streak = 0
        s = "BUY" if action else "SELL"
        # print(f"------------- {datetime.datetime.now().time()} --------------")
        print(f"------------- BTCUSDT: {price}   TIME: {datetime.datetime.now().time()} ----------------")
        print(f"Output: {s}")

        # print(f"Confidence: {abs(output[0][0] - output[0][1])}")
        # print(f"Buy Streak: {buy_streak}")
        # print(f"Sell Streak: {sell_streak}")
        if buy_streak > streak:
            if in_pos == 0:
            #   print("BUY: ", price)
              in_pos = 1
            buy_streak = 0
            #print("Strong Buy")
        if sell_streak > streak:
            if in_pos == 1:  # have a buy
            #   print("SELL: ", price)
              in_pos = 0
            sell_streak = 0
            #print("Strong Sell")
        # print("--------------------------------------------------------------")
    shift_prices = prices[FUTURE_PERIOD_PREDICT:]
    prices = prices[:-FUTURE_PERIOD_PREDICT]
    print("current", prices)
    print("future", shift_prices)
    actual = []
    for i in range(len(shift_prices)):
        if shift_prices[i] > prices[i]:
            actual.append(1)
        else:
            actual.append(0)

    actions = actions[:-FUTURE_PERIOD_PREDICT]
    print("y      : ", actions)
    print("targets: ", actual)

    accuracy = 0
    for i in range(len(actions)):
        if actions[i] == actual[i]:
            accuracy += 1
    accuracy = accuracy/float(len(actions))
    print("accuracy: ", accuracy)


train()
# run()
#test()
# 5:13 pm going down current: $4,937 [[0.6649782  0.33502182]]  -- correct $4,933
# 6:17 down from $4,932 
# 6:35 down  $4,925
# [[0.50210404 0.49789596]]
#[[0.7197066  0.28029343]]
