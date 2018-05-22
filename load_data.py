import numpy as np


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def load_data(raw_values, step=5):
    filtered_values = [float(x) for x in raw_values if isfloat(x)]
    data_x = []
    data_y = []
    for i in range(1, len(filtered_values), step):
        current = float(filtered_values[i])
        sample = []
        for delta in [1, 2, 3, 4, 5, 10, 20, 30, 45, 60, 90, 120, 180, 240, 360, 600, 960, 1440, 2160, 2880, 4320, 5760,
                      7200, 10080]:
            if i - delta < 0:
                sample.append(sample[-1])
            else:
                if isinstance(filtered_values[i - delta], float):
                    sample.append((filtered_values[i - delta] / current) * 100.0)
                else:
                    if len(sample) > 0:
                        sample.append(sample[-1])
                    else:
                        sample.append(100.0)

        sample.reverse()

        for delta in [5, 30, 60, 300, 100]:
            if i + delta >= len(filtered_values):
                sample.append(sample[-1])
            else:
                if isinstance(filtered_values[i + delta], float):
                    sample.append((filtered_values[i + delta] / current) * 100.0)
                else:
                    if len(sample) > 0:
                        sample.append(sample[-1])
                    else:
                        sample.append(100.0)
        # print(sample)
        # print(len(sample))
        data_x.append(sample[:24])
        data_y.append(sample[24:])
    return np.array(data_x), np.array(data_y)
