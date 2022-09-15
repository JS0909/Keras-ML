import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)

x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

###### scale ######
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size(), y_train.size())
# torch.Size([404, 13]) torch.Size([404, 1])

# 2. model
model = nn.Sequential(
    nn.Linear(13, 74),
    nn.ReLU(),
    nn.Linear(74, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 128),
    nn.ReLU(),
    nn.Linear(128, 1),
).to(DEVICE)


# 3. compile, fit
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

EPOCHS = 10000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print(epoch, '\t', loss)

# eval, pred
def evaluate(model, criterion, x_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(x_test)
        loss = criterion(pred, y_test)
    return loss.item()

loss = evaluate(model, criterion, x_test, y_test)
pred_result = model(x_test).cpu().detach().numpy()
score = r2_score(pred_result, y_test.cpu().detach().numpy())

print(f'loss:{loss}')
# print(f'pred_result:{pred_result}')
print(f'score:{score:.4f}')

# loss:33.08406448364258
# score:0.6580