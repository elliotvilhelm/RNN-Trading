from tensorflow.python.client import device_lib

import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import numpy as np
import random

SEQ_LEN = 60 # last 60 minutes
FUTURE_PERIOD_PREDICT= 3 # 3 min
RATIO_TO_PREDICT = "BTC-USD"

def classify(current, future):
    if float(future) > float(current):
        return 1 # BUY else:
    else:
        return 0 # SELL

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

    random.shuffle(sequential_data)
    buys = []
    sells = []
    for seq, target in sequential_data:
        if target == 0:
            sells.append([seq, target])
        elif target == 1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    lower = min(len(buys), len(sells))
    buys = buys[:lower]
    sells = sells[:lower]

    sequential_data = buys + sells
    random.shuffle(sequential_data)
    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), y


def train():
    main_df = pd.DataFrame()
    ratios = ["BTC-USD", "LTC-USD", "ETH-USD"] #, "BCH-USD"]
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



    import time

    EPOCHS = 50  # how many passes through our data
    BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model


    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import ModelCheckpoint

    model = Sequential()
    model.add(LSTM(156, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(156, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(156))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(48, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(2, activation='softmax'))

    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

    # Compile model
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )
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
    score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # Save model
    model.save("models/{}".format(NAME))

def predict(model, observation):
    out = model.predict(observation)
    # [[-2.08416560e-01, -7.64247948e-01,  7.13227434e-01,
    #     -7.93043536e-03,  2.62185234e-01,  2.01831999e-01]]
    # )
    return out

# train()

from collections import deque
from binance.client import Client
from config import api_key, api_secret
import binance_constants

def go_live(model):
    client = Client(api_key, api_secret)
    feed = deque()
    currencies = ["BTCUSDT", "LTCUSDT", "ETHUSDT"] #, "BCH-USD"]

    OPEN_TIME_IDX = 0
    OPEN_IDX = 1
    HIGH_IDX = 2
    LOW_IDX = 3
    CLOSE_IDX = 4
    VOLUME_IDX = 5
    main_data = pd.DataFrame()
    for currency in currencies:
        data = client.get_historical_klines(
            symbol=currency, interval=binance_constants.KLINE_INTERVAL_1MINUTE, start_str="2 hours ago UTC")
        data = list(map(lambda x: [float(x[CLOSE_IDX]), float(x[VOLUME_IDX])], data))  # opentime open high low close volume
        data = pd.DataFrame(data, columns=[f"{currency}_close", f"{currency}_volume"])
        main_data = pd.concat([main_data, data], axis=1)

    df = main_data
    for col in df.columns:
        df = df[df[col] != 0]
        df[col] = df[col].pct_change()  # normalize
        df.dropna(inplace=True)
        df[col] = preprocessing.scale(df[col].values)  # scale between 0 and 1
    df.dropna(inplace=True)
    df = df[-60:]
    test_point = np.array([df.values])
    output = predict(model, test_point)
    return output


def run():
    from tensorflow.keras.models import load_model

    model = load_model(
        "models/RNN_Final-10-0.553.model",
        custom_objects=None,
        compile=True
    )
    import time
    import datetime

    buy_streak = 0
    sell_streak = 0
    while True:
        time.sleep(10)
        output = go_live(model)
        action = output.argmax()
        if action == 0:
            sell_streak += 1
            buy_streak = 0
        else:
            buy_streak += 1
            sell_streak = 0
        print(f"------------- {datetime.datetime.now().time()} --------------")
        print(f"Ouput: {output}")
        print(f"Confidence: {abs(output[0][0] - output[0][1])}")
        print(f"Buy Streak: {buy_streak}")
        print(f"Sell Streak: {sell_streak}")
        if buy_streak > 5:
            print("Strong Buy")
        if sell_streak > 5:
            print("Strong Sell")
        print("--------------------------------------------------------------")

train()
# run()

