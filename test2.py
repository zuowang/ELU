import numpy as np
import matplotlib.pyplot as plt

import math
plt.figure(1)
x = np.linspace(-10, 10, 1000000)
plt.plot(x, 0.5*x*(1+np.tanh(math.sqrt(2/math.pi)*(x+0.044715*x*x*x))))
plt.plot(x, [np.exp(i) - 1 if i <= 0 else i for i in x])
plt.plot(x, [25.6302*(np.exp(0.01*i) - 1) if i <= 0 else i for i in x])
plt.plot(x, [max(0,i) for i in x])
plt.plot(x, np.tanh(x))
plt.plot(x, 1.0/(1+np.exp(-x)))
plt.plot(x, [0.02*i if i <= 0 else i for i in x])
plt.plot(x, x/(1+np.exp(-x)))

plt.show()
