import pandas
import matplotlib.pyplot as pyplot
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.utils.vis_utils import plot_model
from load_data import load_data

btc_dataset = pandas.read_csv('btc-usd.csv')
btc_values = btc_dataset.values[:, 1]



# plt.figure()
# plt.plot(values[:, 1])
# plt.show()


data_x, data_y = load_data(btc_values, 10)

num_sample = len(data_x)
time_steps_x = len(data_x[1])
time_steps_y = len(data_y[1])
print("num sample " + str(num_sample))
print("steps x " + str(time_steps_x))
print("steps y " + str(time_steps_y))

model = Sequential()
model.add(LSTM(time_steps_y, input_shape=(time_steps_x, 1)))
# model.add(TimeDistributed(Dense(time_steps_y)))
model.add(Dense(time_steps_y))
model.compile(loss='mse', optimizer='adam')
# fit network
data_x = np.reshape(data_x, (num_sample, time_steps_x, 1))
# data_y = np.reshape(data_y, (num_sample, time_steps_y, 1))

print(data_x.shape)
print(data_y.shape)

print(model.summary())

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

history = model.fit(data_x, data_y, epochs=5è0, batch_size=1, verbose=2)
# plot history
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')
# pyplot.legend()
# pyplot.show()
# test_x, test_y = load_data(eth_values, 50)
#
# scores = model.evaluate(test_x, test_y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
