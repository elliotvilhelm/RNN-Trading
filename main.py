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

    import pdb; pdb.set_trace()

    train_x, train_y = preprocess_df(main_df)
    validation_x, validation_y = preprocess_df(validation_main_df)
    print(f"train data: {len(train_x)} validation: {len(validation_x)}")
    print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
    print(f"VALIDATION Dont buys: {validation_y.count(0)}, buys: {validation_y.count(1)}")



    import time

    EPOCHS = 10  # how many passes through our data
    BATCH_SIZE = 64  # how many batches? Try smaller batch if you're getting OOM (out of memory) errors.
    NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"  # a unique name for the model


    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import ModelCheckpoint


    model = Sequential()
    model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
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
    import pdb
    pdb.set_trace();
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
from tensorflow.keras.models import load_model

model = load_model(
    "models/RNN_Final-02-0.551.model",
    custom_objects=None,
    compile=True
)
big =   np.array([[[-2.08416560e-01, -7.64247948e-01,  7.13227434e-01,
        -7.93043536e-03,  2.62185234e-01,  2.01831999e-01],
       [ 1.87428461e+00,  1.32760756e+00,  1.97219289e+00,
        -7.59253782e-03,  1.75086792e+00, -3.05533837e-01],
       [-1.68375105e+00,  1.53753632e-01, -1.08424354e+00,
        -7.94807161e-03, -1.83691999e+00, -2.52286533e-03],
       [ 1.26950122e-01, -5.89727889e-01, -5.46402720e-01,
        -7.95967125e-03, -6.13647807e-01, -3.19889830e-01],
       [ 3.70338789e-01,  1.25009920e-01,  1.72700667e-01,
        -7.95184597e-03, -8.82163750e-02, -1.73626079e-01],
       [-1.07362250e+00,  5.56170590e+00, -1.26593433e+00,
        -7.95697724e-03, -1.05216451e+00, -2.65806883e-01],
       [-1.37176221e+00, -1.02697496e+00, -1.87260352e-01,
        -7.92839428e-03, -1.76009004e-01, -6.88903158e-02],
       [ 3.38226478e-01, -3.09762140e-01,  3.53110470e-01,
        -7.96158721e-03,  7.88902232e-01, -3.48134377e-01],
       [ 1.81183632e-01, -5.30669459e-01,  8.93129700e-01,
        -7.27908549e-03,  2.62398307e-01, -4.00635481e-01],
       [ 2.35295767e-01, -9.54913959e-02, -3.66889718e-01,
        -7.96782196e-03,  8.70522071e-02, -3.19317532e-01],
       [-2.68192229e+00,  5.25050399e+00, -1.26675277e+00,
        -4.93714091e-03, -1.49042959e+00,  5.23185369e+00],
       [ 1.79631935e+00, -8.67037842e-01,  8.93911373e-01,
        -7.96632457e-03,  5.25892904e-01, -5.63608321e-01],
       [-1.17782397e+00, -3.53050177e-01, -9.07260082e-01,
        -7.96230567e-03, -4.39134853e-01, -4.07088674e-01],
       [ 1.59658714e-01, -4.66120927e-01, -1.87377628e-01,
        -7.95331279e-03,  4.38113089e-01, -9.81022696e-02],
       [ 4.41339663e-01, -2.89113798e-01, -1.87416754e-01,
        -7.57659637e-03, -1.76009004e-01, -1.23281987e-01],
       [-1.34604363e+00,  3.57328896e-01, -7.16279469e-03,
        -7.96741271e-03, -5.26921105e-01, -1.21127788e-01],
       [ 1.02684334e+00, -4.03786866e-01,  1.73130308e-01,
        -7.78637361e-03,  2.62689419e-01, -4.42068845e-01],
       [ 4.58819636e-02, -2.15641905e-01,  8.94107004e-01,
        -7.96656092e-03,  1.05229989e+00, -3.93554223e-01],
       [-5.12041223e-01, -5.74185203e-01, -9.07455289e-01,
        -7.95880375e-03, -6.14235285e-01,  3.86034192e-02],
       [ 1.09157686e+00,  6.38242839e-01,  1.25461492e+00,
        -7.66448315e-03,  7.01077162e-01, -5.70837475e-02],
       [-1.45413999e+00,  8.82486729e-02, -1.62698669e+00,
        -7.69662095e-03, -1.40310213e+00,  1.14262801e-01],
       [-4.41807726e-01, -7.98855885e-01,  1.73169468e-01,
        -7.96800787e-03,  5.25970608e-01, -3.63524219e-01],
       [-7.67109156e-01,  6.99845419e-01,  1.73130308e-01,
        -6.70662233e-03, -2.63756438e-01,  1.40460460e-01],
       [ 8.21295767e-01, -4.11188095e-01, -7.16279469e-03,
        -7.92426315e-03,  8.71492013e-02,  3.17913873e-01],
       [ 1.14616656e+00, -2.45710861e-01, -7.16279469e-03,
        -7.90402387e-03,  7.89018719e-01, -5.37407607e-01],
       [ 1.86724438e-01, -3.73452501e-01,  1.73091165e-01,
        -7.96146198e-03,  9.63847366e-01, -2.75586901e-01],
       [ 9.77512822e-01,  1.42487374e+00,  3.53266872e-01,
        -7.82812773e-03,  1.74618709e-01,  1.07161681e+00],
       [ 5.38521613e-01, -2.72636420e-01,  7.13383734e-01,
        -7.96139839e-03,  6.12554574e-01, -5.40822942e-01],
       [-4.14219390e-01,  2.28094795e-01, -3.67123661e-01,
        -7.96452297e-03, -6.13422154e-01,  3.86971466e-02],
       [ 3.97734727e-01, -8.19501286e-01,  1.72895704e-01,
        -7.84259745e-03,  4.37369904e-01, -2.22352315e-01],
       [-2.95145996e-01,  6.73839878e-02, -3.67201710e-01,
        -7.94322566e-03,  3.49648586e-01, -4.22175300e-01],
       [ 1.21636004e-01, -9.05241869e-01,  1.72934762e-01,
        -7.96249200e-03,  3.49545468e-01, -9.05506910e-02],
       [ 5.27600795e-01,  5.47185174e-01, -7.16279469e-03,
        -7.96725001e-03,  3.49442411e-01,  4.06210629e-02],
       [ 3.70529119e-01,  2.28228308e+00, -7.16279469e-03,
        -7.74976186e-03,  1.74373822e-01, -2.55352325e-01],
       [-1.05671055e-01, -9.49452437e-01,  5.33012702e-01,
        -7.95951087e-03,  1.74348084e-01, -3.46908522e-01],
       [-7.33336953e-01, -1.62462153e-01, -7.26928498e-01,
        -7.76106906e-03, -8.75162393e-01, -2.84094717e-01],
       [ 1.12298852e+00,  9.37503760e-02,  5.33129877e-01,
        -7.96502794e-03,  7.87101061e-01,  9.87914062e-01],
       [-1.23090247e+00, -8.19521558e-02,  5.32778505e-01,
        -7.54683832e-03, -8.80552648e-02, -5.21780896e-01],
       [-2.51869896e-01,  3.12117499e-01, -7.26616641e-01,
        -7.95829351e-03, -3.50471480e-01, -2.87639557e-01],
       [-3.53518556e-02, -1.88243867e-01,  3.52876120e-01,
        -7.96300337e-03,  2.61895236e-01,  1.30242594e-01],
       [-5.15937211e-02,  1.43686283e-02, -1.87104220e-01,
        -7.93451842e-03, -1.75544493e-01, -3.35279967e-01],
       [-3.06048764e-01, -5.59888481e-01, -7.16279469e-03,
        -7.96071897e-03,  4.36854390e-01,  1.84236916e-01],
       [-3.53578906e-02, -4.55789522e-02, -3.67123661e-01,
        -6.07358927e-03, -4.37877082e-01, -1.80010551e-01],
       [ 1.13423647e+00, -3.92066860e-01,  1.07318820e+00,
        -7.96507976e-03,  8.74300551e-01, -2.52899039e-01],
       [ 7.00666561e-01, -4.59239505e-01,  1.72661710e-01,
        -7.88874645e-03, -8.74840930e-01, -3.60402351e-01],
       [-2.86965438e-03,  3.61021814e-01, -1.86948359e-01,
        -7.96389990e-03,  5.24343622e-01, -3.13189378e-01],
       [-1.04158536e+00, -4.42279670e-01, -1.62558334e+00,
        -7.66685652e-03, -1.22490061e+00,  1.39656736e-01],
       [-2.51853613e-01, -4.38749694e-01, -5.47689968e-01,
        -7.96294277e-03,  7.87275008e-01, -4.27563538e-01],
       [-2.51883576e-01,  4.51670343e-01,  1.73130308e-01,
        -7.96064828e-03, -5.25488548e-01, -1.20692933e-01],
       [ 3.50283292e-02, -4.92194054e-01, -7.16279469e-03,
        -7.95995381e-03, -5.25720325e-01,  1.70894383e+00],
       [-1.16561523e-01, -4.88489632e-01,  5.33599084e-01,
        -7.94776865e-03,  1.74528408e-01, -5.19558269e-01],
       [ 1.52393361e+00,  6.47063270e-01,  1.07365700e+00,
        -6.72979084e-03,  1.22506900e+00, -2.18180278e-01],
       [ 3.97485438e-01, -1.89177639e-01,  5.32544511e-01,
        -7.95109235e-03,  1.74322354e-01,  1.09138802e-01],
       [-4.57238873e-01, -3.23022648e-01, -9.06090619e-01,
        -7.96646657e-03, -3.50368574e-01, -4.63074413e-01],
       [-2.57158548e-01,  8.43063425e-02, -1.87143228e-01,
        -7.95793989e-03, -4.37941407e-01, -2.61721337e-01],
       [ 5.00359574e-01, -5.24588195e-01,  3.52876120e-01,
        -7.95691871e-03,  1.74412443e-01, -1.90946972e-01],
       [ 5.70563986e-01,  2.30933455e-01,  3.52720057e-01,
        -7.32240229e-03,  6.99322086e-01,  8.48091465e-01],
       [-3.65222575e-01, -6.35466884e-01, -7.26616641e-01,
        -7.95434962e-03, -5.25218400e-01, -5.28768041e-01],
       [-4.35605637e-01, -2.37751678e-01, -7.16279469e-03,
        -7.96304465e-03, -2.63020854e-01, -6.83689634e-02],
       [-1.97641580e-01, -5.53530541e-02, -5.47221167e-01,
        -7.94495500e-03, -4.38070115e-01, -2.61359499e-01]]])

# prob = predict(model, 0)
# print(prob)

from collections import deque
from binance.client import Client
from config import api_key, api_secret
import binance_constants

def go_live():
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
    action = output.argmax()
    print(output, action)

import time
while True:
    print("------------------")
    time.sleep(2)
    go_live()

