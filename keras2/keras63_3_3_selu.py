import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.activations import selu

x = np.arange(-5, 5, 0.1)
y = selu(x)

plt.plot(x, y)
plt.grid()
plt.show()
