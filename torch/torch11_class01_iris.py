# 회귀에 0과 1의 논리를 붙인 이진분류 모델 (회귀에 sigmoid 붙인 것)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data
datasets = load_iris()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
# y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
# y_train = torch.FloatTensor(y_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)

x_test = torch.FloatTensor(x_test)
# y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)
# y_test = torch.FloatTensor(y_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

###### scale ######
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())
# torch.Size([150, 4])

# 2. model
class Model(nn.Module): # 상속은 상위 클래스만 넣을 수 있음
    def __init__(self, input_dim, output_dim): # 사용할 레이어들 정의
        # super().__init__() # Model의 부모 클래스도 init 해줌
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, input_size): # 실제 모델 구성
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
    
model = Model(4, 3).to(DEVICE)

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

# loss:0.1816771775484085
# pred_result:tensor([1, 0, 2, 2, 0, 0, 2, 1, 2, 0, 0, 2, 2, 1, 2, 1, 0, 0, 0, 0, 0, 2, 2, 1,
#         2, 2, 1, 1, 1, 1], device='cuda:0')
# score:0.9667
# acc_score:0.9667