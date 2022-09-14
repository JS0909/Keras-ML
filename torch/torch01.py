import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 1. data
x = np.array([1,2,3]) # (3, )
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1) # 1번째자리 쉐이프 늘려주기, 0이면 0번째, -1 마지막 번째
y = torch.FloatTensor(y).unsqueeze(-1) # 만약 2라면 (3,1,1)이 된다

print(x, y)
print(x.shape, y.shape) # torch.Size([3, 1]) torch.Size([3, 1])

# 2. model
# model = Sequnential()
model = nn.Linear(1, 1) # 인풋 열, 아웃풋 열 / 단층레이어

# 3. compile, fit
# model.compile(loss='mse', optimizer='SGD')
criterion = nn.MSELoss() # = loss
optimizer = optim.SGD(model.parameters(), lr=0.001) # 정의한 모델의 파라미터에 러닝레이트를 적용한 옵티마이저를 쓰겠다
# optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad() # 손실함수 기울기를 0으로 초기화. 두번째 역전파부터 이전 grad가 남아 있으므로 0으로 한번씩 초기화해줘야 함
    
    hypothesis = model(x)
    
    
    loss.backward()  # 역전파하겠다
    optimizer.step() # 역전파한거 가지고 가중치를 갱신시키겠다

# 4. eval




