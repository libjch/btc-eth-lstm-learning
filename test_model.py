import pandas
import matplotlib.pyplot as pyplot
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model
from keras.models import model_from_json
import os
from load_data import load_data


def test_model(model):
    eth_dataset = pandas.read_csv('eth-usd-small.csv')
    btc_values = eth_dataset.values[:, 1]
    test_x, test_y = load_data(btc_values, 5)

    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    predicted = model.predict(test_x)

    print("Predicted:")
    print(predicted[:20])
    print("Values:")
    print(test_y[:20])
    try:
        pyplot.figure(figsize=(800, 800))
        pyplot.plot(test_y[:10], color='black')
        pyplot.plot(predicted[:10], color='blue')
        pyplot.show()
    except Exception as e:
        print(e)

    # print("%s: %.2f%%" % (loaded_model.metrics_names[0], score[0] * 100))
