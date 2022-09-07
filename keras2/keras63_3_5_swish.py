import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.activations import swish

x = np.arange(-5, 5, 0.1)
y = swish(x)

plt.plot(x, y)
plt.grid()
plt.show()
