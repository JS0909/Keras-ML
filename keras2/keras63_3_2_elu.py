import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.activations import elu

def elu1(x, a):
    return np.maximum(0, x) + np.maximum(x, a*(np.exp(x)-1))

elu2 = lambda x, a: (x<0)*a*(np.exp(x)-1) + (x>=0)*x

x = np.arange(-5, 5, 0.1)
y = elu1(x, 0.1)

plt.plot(x, y)
plt.grid()
plt.show()
