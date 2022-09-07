import numpy as np

f = lambda x: x**2 - 4*x + 6
# def f(x):
#     return x**2 - 4*x + 6

gradient = lambda x: 2*x - 4

x = 50.0       # 초기값
epochs = 20
learning_rate = 0.25

print('step\t x\t f(x)')
print('{:02d}\t {:6.5f}\t {:6.5}\t'.format(0, x, f(x)))

for i in range(epochs):
    x = x - learning_rate * gradient(x)
    print('{:02d}\t {:6.5f}\t {:6.5}\t'.format(i+1, x, f(x)))


# x 갱신