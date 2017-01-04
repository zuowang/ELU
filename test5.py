import numpy as np
import pickle
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages

def get_median_curve(data, mode='train', metric='loss', start_key=1, number_to_merge=5, decay=0.9):
    curves = []
    for i in range(start_key, start_key + number_to_merge):
        unsmoothed = data[mode + '_' + metric + str(i)]

        ema = unsmoothed[0]
        ema_list = []
        for pt in unsmoothed:
            ema = decay * ema + (1 - decay) * pt
            ema_list.append(ema)
        curves.append(ema_list[:])

    return np.median(curves, axis=0)

def straight_line(d):
    data = d.copy()
    flag = False
    min_data = np.min(data)
    for i in range(len(data)):
        if abs(data[i] - min_data) < 1e-9:
            flag = True
        if flag:
            data[i] = min_data
    return data



#gelu_data = pickle.load(open("C:/wanda/mnist_fcn_gelu.p", "rb"))
#elu_data = pickle.load(open("C:/wanda/mnist_fcn_elu.p", "rb"))
silu_data = pickle.load(open("C:/wanda/mnist_fcn_silu.p", "rb"))
selu_data = pickle.load(open("C:/wanda/mnist_fcn_selu.p", "rb"))
#soi_data = pickle.load(open("C:/wanda/mnist_fcn_soi.p", "rb"))



decay = 0.995
start = 1
first_point = 0
#data = get_median_curve(gelu_data, decay = decay, number_to_merge=5, start_key=start, mode='train')
plt.figure(figsize=(10,10))
#plt.plot(np.linspace(0, 50, len(data[first_point:])), data[first_point:])
#data = get_median_curve(elu_data, decay = decay, number_to_merge=5, start_key=start, mode='train')
#plt.plot(np.linspace(0, 50, len(data[first_point:])), data[first_point:])
data = get_median_curve(silu_data, decay = decay, number_to_merge=5, start_key=start, mode='train')
plt.plot(np.linspace(0, 50, len(data[first_point:])), data[first_point:])
data = get_median_curve(selu_data, decay = decay, number_to_merge=5, start_key=start, mode='train')
plt.plot(np.linspace(0, 50, len(data[first_point:])), data[first_point:])
#data = get_median_curve(soi_data, decay = decay, number_to_merge=5, start_key=start, mode='train')
#plt.plot(np.linspace(0, 50, len(data[first_point:])), data[first_point:])
plt.ylim((0,.15))
plt.legend(['gelu', 'elu', 'silu', 'selu', 'soi'])

plt.show()
