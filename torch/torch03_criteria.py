import numpy as np
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()                   
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, '사용DVICE:', DEVICE)

# 1. data
x = np.array([1,2,3]) # (3, )
y = np.array([1,2,3])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE) # (3, 1)
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)

# 2. model
model = nn.Linear(1, 1).to(DEVICE) 

# 3. compile, fit
criterion = nn.MSELoss() # = loss
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x, y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss().forward(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y)
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward() 
    optimizer.step()

    return loss.item()
    
epochs = 20000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
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

results = model(torch.Tensor([[4]]).to(DEVICE))
print(f'4의 예측값: {results.item()}')