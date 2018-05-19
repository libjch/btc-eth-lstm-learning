import pandas
import matplotlib.pyplot as pyplot
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed

dataset = pandas.read_csv('btc-usd-small.csv')

values = dataset.values[:, 1]
# plt.figure()
# plt.plot(values[:, 1])
# plt.show()


data_x = []
data_y = []
for i in range(1, len(values), 5):
    current = values[i]
    sample = []
    for delta in [1, 2, 3, 4, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 600, 960, 1440, 2160, 2880, 4320, 5760,
                  7200, 10080]:
        if i - delta < 0:
            sample.append(sample[-1])
        else:
            sample.append(values[i - delta] / current)
    sample.reverse()

    for delta in [5, 30, 60, 300, 100, 3600]:
        if i + delta > len(values):
            sample.append(sample[-1])
        else:
            sample.append(values[i + delta] / current)
    # print(sample)
    # print(len(sample))
    data_x.append(sample[:24])
    data_y.append(sample[24:])

data_x = np.array(data_x)
data_y = np.array(data_y)

num_sample = len(data_x)
time_steps_x = len(data_x[1])
time_steps_y = len(data_y[1])
print("num sample " + str(num_sample))
print("steps x " + str(time_steps_x))
print("steps y " + str(time_steps_y))

model = Sequential()
model.add(
    LSTM(time_steps_y, input_shape=(time_steps_x, 1)))
# model.add(TimeDistributed(Dense(time_steps_y)))
model.add(Dense(time_steps_y))
model.compile(loss='mse', optimizer='adam')
# fit network
data_x = np.reshape(data_x, (num_sample, time_steps_x, 1))
# data_y = np.reshape(data_y, (num_sample, time_steps_y, 1))

print(data_x.shape)
print(data_y.shape)

print(model.summary())

history = model.fit(data_x, data_y, epochs=50, batch_size=1, verbose=2)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
