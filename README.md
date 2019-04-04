#### Training At: http://75.80.54.222:6006
#### Data Age: 03/31/2019 @ 10:08am (UTC)

# RNN Bitcoin Movement Prediction
This project aims to predict Bitcoin price movement over short tiem intervals. The current best model has a validation accuracy of 66.9% predicting price change 120 minutes into the future. The current models base coin is Tether (USDT). I chose this assest to avoid volatility in our base currency. The data for the network is generated using a tool I wrote for the Binance API located at https://github.com/ElliotVilhelm/Binance-Exchange-to-CSV. The tool is used to collect the `Close` and `Volume` of the 4 coins being exchanged for Tether at the highest rate. Additional data is generated using technical indicators on the `BTCUSDT` data and joined with the existing dataset.

Normilzation is applied via `pct_change()` to all columns other than `BTC-USDT_TSI` and `BTC-USDT_MOMENTUM`. Scaling is applied to all columns. A validation split of 5% was used for testing. The validation split was done using disjoint sequences rather than a continuous segment. This is done with hopes of promoting generalization.

##### Input
- BTC-USDT Close
- BTC-USDT Volume
- BTC-USDT Moving Average
- BTC-USDT Exponential Moving Average
- BTC-USDT True Strength Index
- BTC-USDT Momentum
- ETH-USDT Close
- ETH-USDT Volume
- LTC-USDT Close
- LTC-USDT Volume
- EOS-USDT Close
- EOS-USDT Volume
- BCHABC-USDT Close
- BCHABC-USDT Volume

##### Output
- Probability of `BTCUSDT` Price movement `UP`
- Probability of `BTCUSDT` Price movement `DOWN`


### Architecture
```
Layer (type)                 Output Shape              Param #   
=================================================================
lstm (LSTM)                  (None, 180, 128)          73216     
_________________________________________________________________
dropout (Dropout)            (None, 180, 128)          0         
_________________________________________________________________
batch_normalization (BatchNo (None, 180, 128)          512       
_________________________________________________________________
lstm_1 (LSTM)                (None, 180, 256)          394240    
_________________________________________________________________
dropout_1 (Dropout)          (None, 180, 256)          0         
_________________________________________________________________
batch_normalization_1 (Batch (None, 180, 256)          1024      
_________________________________________________________________
lstm_2 (LSTM)                (None, 128)               197120    
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 128)               512       
_________________________________________________________________
dense (Dense)                (None, 64)                8256      
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 130       
=================================================================
Total params: 675,010
Trainable params: 673,986
Non-trainable params: 1,024
_________________________________________________________________
```

### Insights
- Recent data is much more relevant, e.g. looking at the past 20 minutes for a prediction 10 minutes ahead is much better than looking at the past 120 minutes for the same prediction.


# Resources
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/

https://danijar.com/tips-for-training-recurrent-neural-networks/
