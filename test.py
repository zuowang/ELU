import numpy as np
import matplotlib.pyplot as plt
import pickle

plt.figure(1)
x = np.linspace(1, 50, 21450)
history1 = pickle.load(open("C:/wanda/mnist_fcn_selu.p", "rb"))
history2 = pickle.load(open("C:/wanda/mnist_fcn_gelu.p", "rb"))
y2 = [np.median([history2["train_loss1"][i],history2["train_loss2"][i],history2["train_loss3"][i], history2["train_loss4"][i],history2["train_loss5"][i]]) for
 i in range(21450)]
plt.plot(x, y2,'r')
y1 = [np.median([history1["train_loss1"][i],history1["train_loss2"][i],history1["train_loss3"][i], history1["train_loss4"][i],history1["train_loss5"][i]]) for
 i in range(21450)]
plt.plot(x, y1,'b')
plt.legend(['gelu', 'selu', 'silu', 'selu', 'soi'])



plt.show()
