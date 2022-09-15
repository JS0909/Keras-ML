import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('[cuda:0, cuda:1]' if USE_CUDA else 'cpu') # 리스트 형태로 여러 gpu 사용 가능
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data
datasets = load_digits()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train).to(DEVICE)

x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test).to(DEVICE)

###### scale ######
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size(), len(y_train.unique()))
# torch.Size([1437, 64]) 10

# 2. model
model = nn.Sequential(
    nn.Linear(64, 74),
    nn.ReLU(),
    nn.Linear(74, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 128),
    nn.ReLU(),
    nn.Linear(128, 3),
).to(DEVICE)


# 3. compile, fit
criterion = nn.CrossEntropyLoss() # softmax + sparse_categorical_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, x_train, y_train):
    model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

EPOCHS = 1000
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
pred_result = torch.argmax(model(x_test), 1)

score = (pred_result == y_test).float().mean()
acc_score = accuracy_score(y_test.cpu(), pred_result.cpu())

print(f'loss:{loss}')
print(f'pred_result:{pred_result}')
print(f'score:{score:.4f}')
print(f'acc_score:{acc_score:.4f}')

# 터짐