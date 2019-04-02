def classify(current, future):
    if float(future) > float(current):
        return 1 # BUY else:
    else:
        return 0 # SELL