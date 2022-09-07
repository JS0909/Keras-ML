import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)
    
relu2 = lambda x: np.maximum(0, x)

x = np.arange(-5, 5, 0.1)
y = relu2(x)

plt.plot(x, y)
plt.grid()
plt.show()
