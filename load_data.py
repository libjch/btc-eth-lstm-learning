import numpy as np


def load_data(raw_values, step=5):
    data_x = []
    data_y = []
    for i in range(1, len(raw_values), step):
        current = raw_values[i]
        sample = []
        for delta in [1, 2, 3, 4, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 600, 960, 1440, 2160, 2880, 4320, 5760,
                      7200, 10080]:
            if i - delta < 0:
                sample.append(sample[-1])
            else:
                if isinstance(raw_values[i - delta], float):
                    sample.append(raw_values[i - delta] / current)
                else:
                    if len(sample) > 0:
                        sample.append(sample[-1])
                    else:
                        sample.append(1)

        sample.reverse()

        for delta in [5, 30, 60, 300, 100, 3600]:
            if i + delta > len(raw_values):
                sample.append(sample[-1])
            else:
                if isinstance(raw_values[i + delta], float):
                    sample.append(raw_values[i + delta] / current)
                else:
                    if len(sample) > 0:
                        sample.append(sample[-1])
                    else:
                        sample.append(1)
        # print(sample)
        # print(len(sample))
        data_x.append(sample[:24])
        data_y.append(sample[24:])
    return np.array(data_x), np.array(data_y)
