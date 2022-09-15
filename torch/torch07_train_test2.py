import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 1. data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

x = torch.Tensor(x).unsqueeze(-1).to(DEVICE)
y = torch.Tensor(y).unsqueeze(-1).to(DEVICE)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.85, shuffle=True, random_state=1234)

### scaling ###
x_test = (x_test - torch.mean(x_train)) / torch.std(x_train)
x_train = (x_train - torch.mean(x_train)) / torch.std(x_train)

print(x_train.shape, y_train.shape)

# 2. model
model = nn.Sequential(
    nn.Linear(1, 64),
    nn.SELU(),
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Linear(64, 1)
).to(DEVICE)

# compile, fit
optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    print(epoch, '\t', loss)
    
# eval, pred
def evaluate(model, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        pred_y = model(x_test)
        result = nn.MSELoss()(pred_y, y_test)
        
        return result.item()

loss = evaluate(model, x_test, y_test)
result = model(x_test).cpu().detach().numpy()

print(f'loss:{loss}')
print(f'예측:{result}')


# loss:1.942571543622762e-05
# 예측:[[8.001041 ]
#  [2.9938545]]