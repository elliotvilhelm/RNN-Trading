#### Training At: [75.80.54.222:6006](75.80.54.222:6006)

# RNN Bitcoin Movement Prediction
This project aims to predict Bitcoin price movement over short intervals. The current best model has a validation accuracy of 66.9% predicting price change 120 minutes into the future.

Input
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

Output
- Probability of Price movement up
- Probability of Price movement down


### Architectures
- Stacked LSTM

### Insights
- Recent data is much more relevant, e.g. looking at the past 20 minutes for a prediction 10 minutes ahead is much better than looking at the past 120 minutes for the same prediction.
- Low dropout near input layers. Less dropout for larger layers.


# Resources
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/
https://pythonprogramming.net/crypto-rnn-model-deep-learning-python-tensorflow-keras/