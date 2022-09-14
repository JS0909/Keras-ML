import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)

# 1. data
x = np.array([[1,2,3,4,5,6,7,8,9,10], [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]]) 
y = np.array([11,12,13,14,15,16,17,18,19,20])
x_test = np.array([10, 1.3])

x = torch.FloatTensor(np.transpose(x)).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(np.transpose(x_test)).to(DEVICE)

###### 스케일링 ######
x_test = (x_test - torch.mean(x)) / torch.std(x) 
x = (x - torch.mean(x)) / torch.std(x) 

print(x, y)

# 2. model
model = nn.Sequential(
    nn.Linear(2,4),
    nn.Linear(4,5),
    nn.Linear(5,3),
    nn.Linear(3,2),
    nn.Linear(2,1)
).to(DEVICE)

# 3. compile, fit
criterion = nn.MSELoss() # = loss
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, optimizer, x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward() 
    optimizer.step()

    return loss.item()
    
epochs = 10000
for epoch in range(1, epochs+1):
    loss = train(model, optimizer, x, y)
    print('epoch: {}, loss: {}'.format(epoch, loss))
    
# 4. eval
def evaluate(model, criterion, x, y): 
    model.eval() 
    
    with torch.no_grad(): 
        x_predict = model(x)
        results = criterion(x_predict, y)
    
    return results.item()

result_loss = evaluate(model, criterion, x, y)
print(f'최종 loss: {result_loss}')

results = model(x_test)
print(f'예측값: {results.item()}')