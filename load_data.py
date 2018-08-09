import numpy as np


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def average_value(data,index,delta):
    if(delta < 3):
        return data[index]
    number = 0.0
    total  = 0.0

    for i in range(0,delta,int(delta/10)+1):
        total += data[index+i] + data[index-i]
        number += 2
    return total / float(number);



def load_data(raw_values, step=5):
    float_values = [float(x) for x in raw_values if isfloat(x)]
    data_x = []
    data_y = []
    for i in range(len(float_values)-1-320,14*24*60 + 1008, -step):
        current = float_values[i]
        sample = []
        for delta in [1, 2, 3, 4, 5, 10, 20, 30, 45, 60, 90, 2*60, 3*60, 4*60, 6*60, 10*60, 16*60, 24*60, 36*60, 2*24*60, 3*24*60, 4*24*60, 5*24*60, 6*24*60,8*24*60,11*24*60,14*24*60]:
            if i - delta < 0:
                sample.append(sample[-1])
            else:
                if isinstance(float_values[i - delta], float):
                    value = average_value(float_values,i-delta, int(delta/20))
                    ratio = (value / current)
                    sample.append(ratio * ratio * 100.0)
                else:
                    if len(sample) > 0:
                        sample.append(sample[-1])
                    else:
                        sample.append(100.0)

        sample.reverse()

        for delta in [1,3,5,30,60,300]:
            if i + delta >= len(float_values):
                sample.append(sample[-1])
            else:
                if isinstance(float_values[i + delta], float):
                    value = average_value(float_values, i + delta, int(delta / 20))
                    ratio = (value / current)
                    sample.append(ratio * ratio * 100.0)
                else:
                    if len(sample) > 0:
                        sample.append(sample[-1])
                    else:
                        sample.append(100.0)
        # print(sample)
        # print(len(sample))
        data_x.append(sample[:27])
        data_y.append(sample[27:])
    return np.array(data_x), np.array(data_y)
