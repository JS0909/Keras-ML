import numpy as np

# 시계열 데이터 자르는 함수

a = np. array(range(1,11)) # 1~10까지의 데이터
size = 5 # 5개씩 자르겠다

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset) - size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset) # 하나하나 리스트 형태로 추가함
    return np.array(aaa)

bbb = split_x(a, size)
print(bbb)
print(bbb.shape) # (6, 5)

x =  bbb[:, :-1]
y =  bbb[:, -1]
print(x, y)
print(x.shape, y.shape) # (6, 4) (6,)
