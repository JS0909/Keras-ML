import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 1. 데이터
# x = np.array([1,2,3,4,5,6,7,8,9,10])
# y = np.array([1,2,3,4,5,6,7,8,9,10])

x_train = np.array([1,2,3,4,5,6,7]) # (7,)
x_test = np.array([8,9,10])         # (3,)
y_train = np.array([1,2,3,4,5,6,7]) # (7,)
y_test = np.array([8,9,10])         # (3,)

x_predict = np.array([11,12,13])

x_train = torch.Tensor(x_train).unsqueeze(-1).to(DEVICE)
x_test = torch.Tensor(x_test).unsqueeze(-1).to(DEVICE)
y_train = torch.Tensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.Tensor(y_test).unsqueeze(-1).to(DEVICE)
x_predict = torch.Tensor(x_predict).unsqueeze(-1).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_predict.shape)
# torch.Size([7, 1]) torch.Size([7, 1]) torch.Size([3, 1])

# keras는 벡터형태의 연산이 가능하지만 torch는 텐서로 보내기때문에 안됨

#### 스케일링 ####
x_predict = (x_predict - torch.mean(x_train)) / torch.std(x_train)
x_test = (x_test - torch.mean(x_train)) / torch.std(x_train)
x_train = (x_train - torch.mean(x_train)) / torch.std(x_train)

# 2. 모델
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 1),
).to(DEVICE)

# 3. 컴파일, 훈련
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = nn.MSELoss()(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss: {loss}')
    
# 4. 평가, 예측
def evaluate(model, x, y):
    model.eval()
    
    with torch.no_grad():
        pred_y = model(x)
        result = nn.MSELoss()(pred_y, y)
        
    return result.item()

loss = evaluate(model, x_test, y_test)
print(f'최종 loss: {loss}')

result = model(x_predict).cpu().detach().numpy()
print(f'예측값: {result}')

# 최종 loss: 0.059064146131277084
# 예측값: [[10.553898]
#  [11.442092]
#  [12.32498 ]]