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

eth_dataset = pandas.read_csv('eth-usd.csv')
eth_values = eth_dataset.values[:, 1]

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mse', optimizer='adam')

data_x, data_y = load_data(eth_values, 50)
data_x = np.reshape(data_x, (data_x.shape[0], data_x.shape[1], 1))

score = loaded_model.evaluate(data_x, data_y, verbose=0)

print(score*100)

# print("%s: %.2f%%" % (loaded_model.metrics_names[0], score[0] * 100))