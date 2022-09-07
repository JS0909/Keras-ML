import numpy as np
import matplotlib.pyplot as plt

def leaky_relu(x, a):
    return np.maximum(a*x, x)
    
leaky_relu2 = lambda x, a: np.maximum(a*x, x)

x = np.arange(-5, 5, 0.1)
y = leaky_relu(x, 0.1)

plt.plot(x, y)
plt.grid()
plt.show()
