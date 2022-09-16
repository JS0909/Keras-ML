import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

USE_CUDA = torch.cuda.is_available()
# DEVICE = torch.device('[cuda:0, cuda:1]' if USE_CUDA else 'cpu') # 리스트 형태로 여러 gpu 사용 가능
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print(torch.__version__, DEVICE) # 1.12.1 cuda:0

# 1. data
datasets = fetch_covtype()
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
# torch.Size([464809, 54]) 7

train_set = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_set, batch_size=1000, shuffle=True)
test_set = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_set, batch_size=1000, shuffle=True)

# 2. model
class Model(nn.Module): # 상속은 상위 클래스만 넣을 수 있음
    def __init__(self, input_dim, output_dim): # 사용할 레이어들 정의
        # super().__init__()
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
    
model = Model(54, 8).to(DEVICE)


# 3. compile, fit
criterion = nn.CrossEntropyLoss() # softmax + sparse_categorical_crossentropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, criterion, optimizer, loader):
    model.train()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

EPOCHS = 1000
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch%20==0:
        print(epoch, '\t', loss)

# eval, pred
def evaluate(model, criterion, loader):
    model.eval()
    
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.item()
            
    return total_loss / len(loader)

loss = evaluate(model, criterion, test_loader)
pred_result = torch.argmax(model(x_test), 1)

score = (pred_result == y_test).float().mean()
acc_score = accuracy_score(y_test.cpu(), pred_result.cpu())

print(f'loss:{loss}')
# print(f'pred_result:{pred_result}')
print(f'score:{score:.4f}')
print(f'acc_score:{acc_score:.4f}')

# loss:0.2606202478592212
# score:0.8969
# acc_score:0.8969