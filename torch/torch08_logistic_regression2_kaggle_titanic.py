import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd


from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data

# 1. 데이터
path = 'd:/study_data/_data/kaggle_titanic/'
train_set = pd.read_csv(path+'train.csv')
test_set = pd.read_csv(path+'test.csv')

train_set = train_set.drop(columns='Cabin', axis=1) 
train_set['Age'].fillna(train_set['Age'].mean(), inplace=True) 
print(train_set['Embarked'].mode()) 
train_set['Embarked'].fillna(train_set['Embarked'].mode()[0], inplace=True) 
train_set.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

y = train_set['Survived']
x = train_set.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
x = x.values
y = np.array(y).reshape(-1, 1)

print(x.shape, y.shape) # (891, 7) (891, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).to(DEVICE)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())

# 2. model
model = nn.Sequential(
    nn.Linear(7, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.ReLU(),
    nn.Linear(64, 128),
    nn.Linear(128, 64),
    nn.Sigmoid(),
    nn.Linear(64, 16),
    nn.Linear(16, 1),
    nn.Sigmoid()
).to(DEVICE)


# 3. compile, fit
criterion = nn.BCELoss() # binary_crossentropy
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
pred_result = (model(x_test) >= 0.5).float()

score = (pred_result == y_test).float().mean()
acc_score = accuracy_score(y_test.cpu(), pred_result.cpu())

print(f'loss:{loss}')
print(f'pred_result:{pred_result}')
print(f'score:{score:.4f}')
print(f'acc_score:{acc_score:.4f}')

# score:0.7877
# acc_score:0.7877