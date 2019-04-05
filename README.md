#### Training At: http://75.80.54.222:6006
#### Data Age: 03/31/2019 @ 10:08am (UTC)

# RNN Bitcoin Movement Prediction
This project aims to predict Bitcoin price movement over short tiem intervals. The current best model has a validation accuracy of 66.9% predicting price change 120 minutes into the future. The current models base coin is Tether (USDT). I chose this assest to avoid volatility in our base currency. The data for the network is generated using a tool I wrote for the Binance API located at https://github.com/ElliotVilhelm/Binance-Exchange-to-CSV. The tool is used to collect the `Close` and `Volume` of the 4 coins being exchanged for Tether at the highest rate. Additional data is generated using technical indicators on the `BTCUSDT` data and joined with the existing dataset.

Normilzation is applied via `pct_change()` to all columns other than `BTC-USDT_TSI` and `BTC-USDT_MOMENTUM`. Scaling is applied to all columns. A validation split of 5% was used for testing. The validation split was done using disjoint sequences rather than a continuous segment. This is done with hopes of promoting generalization.

### Thoughts on Technical Indicators
You might be thinking "Elliot wtf". Neural networks serve as feature extractors so why generate features yourself? Feeding in technical indicators seems strange when we are feeding in the data the indicators are formed off of. To you I say yeah maybe. I am experimenting and have seen models using 14+ indicators. My largest concern about indicators is the possibility of introducing data leakage. More on that later.

Despite these concerns, in the face of rapid overfitting and a lack of much historical data I have confidence in the benifits of the introduction of technical indicators. Perhaps they belong after the LSTM layers, a direct connection to the final feed forward layer.


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

### Avoiding Overfitting and Other Voodo

#### Insights
- Crypto is volatile. As I write this Bitcoin has rallied to near $5,000 (4/4/19). I firmly believe training data MUST be recent. The model must be updated on recent data. Furthermore, much of these coins are not very old and in their introduction saw volatility which is no longer present. All this to say, it is important to be catious about where we select our data start and end points. Keep your end point recent and your start point as far back without introducing non standard behavior, a mental heuristic. 

- Recent data is much more relevant, e.g. looking at the past 20 minutes for a prediction 10 minutes ahead is much better than looking at the past 120 minutes for the same prediction.

#### Data Leakage
- It is possible there is data leakage through the generation of the techinical indicators and shifting of `future` values for training targets.

https://www.kaggle.com/dansbecker/data-leakage

#### Why Suspicion? 
I see the network overfitting on validation data after introducing multiple indicators. By this I mean I have seen validation accuracy go up along side validation loss. This could mean the model is becoming less certain about correct targets while accurately predicting more targets. Testing live has shown models most accurate on validation data are not the best in practice. The probability distributions from more accurate models is noticably spikier reflecting an overconfident model.


#### Execution
I have not spent the time to make this very convinient for others to use. This needs some automation through scripts. This project uses https://github.com/ElliotVilhelm/Binance-Exchange-to-CSV for data collection. 

1. Clone both Repos.
2. Collect the data from the currencies you want to train.
3. Set hyperparameters in `config.py`.
4. Add `binance_config.py` with your Binance API key and secret.
5. Make sure you have Tensorflow and all relevant packages.
6. `python3 main.py` or `nohup python3 main.py 2>&1 > out_file &` followed by `tail -f out_file`.
7. `tensorboard --logdir=logs`

`test()` this will grab the most recent crypto data and make a prediction. It is a "live" consultant of sorts made up of an enssemble of models.

`train()` GPU training.

# Resources
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/

https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/

https://danijar.com/tips-for-training-recurrent-neural-networks/
