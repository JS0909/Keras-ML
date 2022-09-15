import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data
path = 'D:\study_data\_data\kaggle_bike/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

x = train_set.drop(['count'], axis=1)
y = train_set['count']
x, y= x.values, y.values

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, train_size=0.8, random_state=1234)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)

x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

###### scale ######
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size(), y_train.size())
# torch.Size([8708, 12]) torch.Size([8708, 1])

# 2. model
model = nn.Sequential(
    nn.Linear(12, 74),
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
print(f'score:{score:.4f}')

# loss:1621.0130615234375
# score:0.9490