from itertools import cycle

import pandas
import matplotlib.pyplot as pyplot
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
# from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model
from load_data import load_data
from save_model import save_model
from test_model import test_model
from matplotlib import colors as mcolors
from keras.utils import plot_model

btc_dataset = pandas.read_csv('btc-usd.csv')
btc_values = btc_dataset.values[:, 1]


def load_from_file():
    data_x = np.load("btc-usd-prepared-x.npy")
    data_y = np.load("btc-usd-prepared-y.npy")
    return data_x, data_y

def save_to_file():
    data_x, data_y = load_data(btc_values, 5)
    np.save("btc-usd-prepared-x.npy",data_x)
    np.save("btc-usd-prepared-y.npy",data_y)
    return data_x, data_y

data_x, data_y = load_from_file()


num_sample = len(data_x)
time_steps_x = len(data_x[1])
time_steps_y = len(data_y[1])
print("num sample " + str(num_sample))
print("steps x " + str(time_steps_x))
print("steps y " + str(time_steps_y))

model = Sequential()
model.add(LSTM(time_steps_y, input_shape=(time_steps_x, 1)))
model.add(Dropout(0.2))
model.add(Dense(time_steps_y))
model.compile(loss='mse', optimizer='adam')
# fit network
data_x_3d = np.reshape(data_x, (num_sample, time_steps_x, 1))

SHOW_LINES =50
x_axis = list(range(0, 27))
y_axis = list(range(27, 33))
# pyplot.figure(figsize=(800, 800))

cycol = cycle('bgrcmk')

for index in range(0, num_sample,int(num_sample/SHOW_LINES)):
    color = next(cycol)
    pyplot.plot(x_axis, data_x[index], color=color)
    pyplot.plot(y_axis, data_y[index], color=color)
# pyplot.plot(data_y[-10:][1], color='blue')
pyplot.show()


plot_model(model, to_file='model.png', show_shapes=True)
history = model.fit(data_x_3d, data_y, epochs=10, batch_size=72, verbose=2, validation_split=0.2, shuffle=True)

save_model(model)

#test_model(model)
