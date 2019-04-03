# RNN Bitcoin Movement Prediction
This project aims to predict Bitcoin price movement over short intervals.

Input

- Shout out Sentdex
- Tensorflow stuff

### Architectures Tested
- model.summary() pls

### To Do
- test addition of indicators
- SEQ_LEN average? like beermind, .. input avg at every step? what you got to loose? more currencies?
- modularize
- fill out this readme
- add new features after last LSTM cell

### Insights
- Recent data is much more relevant, e.g. looking at the past 20 minutes for a prediction 10 minutes ahead is much better than looking at the past 120 minutes for the same prediction.
- Low dropout near input layers. Less dropout for larger layers.


# Resources
https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/