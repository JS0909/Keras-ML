# sigmoid

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

sigmoid2 = lambda x: 1 / (1 + np.exp(-x))  # -x제곱 : x제곱의 역수

x = np.arange(-5, 5, 0.1) # -5부터 5까지 0.1간격으로
print(x)
print(len(x)) # 100

y = sigmoid(x)

plt.plot(x, y)
plt.grid()
plt.show()

# sigmoid : 0 ~ 1 사이로