import numpy as np

dataset = [[1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]]

dataset = np.array(dataset) 
print(dataset.shape) # (3, 5)

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column -1
        
        if y_end_number > len(dataset): # y짜르는 곳이 데이터셋 길이 벗어나면 멈춤
            break
        tmp_x = dataset[i:x_end_number, : -1] # 현재반복번째 ~ 현재반복번째+타임스텝까지, 전체를 다 저장
        tmp_y = dataset[x_end_number-1: y_end_number, -1] # 현재반복번째+타임스텝-1 ~ 현재반복번째+y칼럼수(y지정일 수)-1, 제일 마지막 행만 저장
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x, y = split_xy3(dataset, 1, 2)

print(x.shape, y.shape) # (2, 1, 4) (2, 2)
print(x,'\ny: ', y)