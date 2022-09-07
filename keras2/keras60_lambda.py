gradient1 = lambda x : 2*x - 4

x = 3
print(gradient1(x))

# 함수화를 간편하게 할 수 있음

def gradient2(x):
    temp = 2*x - 4
    return temp

x = 3
print(gradient2(x))