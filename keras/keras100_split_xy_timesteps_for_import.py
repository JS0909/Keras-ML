import numpy as np

'''
<<< 시계열데이터 훈련하기 위한 분리 함수 >>>

dataset = 전체 데이터셋
time_steps = 짜르고 싶은 x 행의 개수
y_column = 짜르고 싶은 y 행의 개수 // 짜르고 싶은 일자 수를 한 행으로 가져옴

x 행들을 각각 짤라놓은 y에 비교해가면서 지도학습시킬 수 있다
'''

def split_xy3(dataset, time_steps, y_column):
    x, y = list(), list() # 빈 튜플리스트 생성
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