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
criterion = nn.MSELoss()






