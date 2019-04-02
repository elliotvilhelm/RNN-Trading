import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from config import LEARNING_RATE, DECAY, SEQ_LEN, INPUT_DIM

def build_model(loss, opt):
    model = Sequential()
    model.add(LSTM(512, input_shape=(SEQ_LEN, INPUT_DIM), return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(LSTM(512))
    model.add(Dropout(0.25))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    # model.add(Dropout(0.1))

    model.add(Dense(2, activation='softmax'))


    # Compile model
    model.compile(
        loss=loss,
        optimizer=opt,
        metrics=['accuracy']
    )
    return model